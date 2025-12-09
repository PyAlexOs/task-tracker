import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import aio_pika
import aioboto3
from dotenv import load_dotenv

from .core import SpeechTranscriptor

# Конфигурация
RABBIT_URL = "amqp://admin:admin@0.0.0.0:5672/"
TASK_QUEUE = "audio_tasks"
STATUS_QUEUE = "audio_status"
CANCEL_QUEUE = "audio_cancel"
RESULT_QUEUE = "audio_results"

MINIO_ENDPOINT = "http://0.0.0.0:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "adminpassword"
MINIO_BUCKET = "audio-files"
RESULT_BUCKET = "audio-results"


class S3Handler:
    """Хелпер для работы с S3/MinIO."""
    
    def __init__(self) -> None:
        self.session = aioboto3.Session()
        self.endpoint = MINIO_ENDPOINT
        self.access_key = MINIO_ACCESS_KEY
        self.secret_key = MINIO_SECRET_KEY
    
    async def download_file(self, bucket: str, key: str, local_path: Path) -> None:
        """Скачивает файл из S3 в локальную директорию."""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as s3_client:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                data = await stream.read()
                local_path.write_bytes(data)
    
    async def upload_json(self, bucket: str, key: str, data: dict[str, Any]) -> None:
        """Загружает JSON результат в S3."""
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        ) as s3_client:
            # Создаем бакет если не существует
            try:
                await s3_client.head_bucket(Bucket=bucket)
            except Exception:
                await s3_client.create_bucket(Bucket=bucket)
            
            json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            await s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_bytes,
                ContentType="application/json",
            )


class AudioWorker:
    def __init__(self, transcriptor: SpeechTranscriptor) -> None:
        """Конструктор класса.

        Args:
            transcriptor (SpeechTranscriptor): Транскрибатор-диаризатор для обработки задач.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.transcriptor = transcriptor
        self.s3_handler = S3Handler()

        self.connection: aio_pika.RobustConnection | None = None
        self.channel: aio_pika.RobustChannel | None = None

        self.task_queue: aio_pika.Queue | None = None
        self.status_exchange: aio_pika.Exchange | None = None
        self.cancel_queue: aio_pika.Queue | None = None
        self.result_queue: aio_pika.Queue | None = None

        # Управление одной задачей
        self.current_task_id: str | None = None
        self.current_task_future: asyncio.Task | None = None
        self.cancel_event: asyncio.Event = asyncio.Event()

    async def connect(self) -> None:
        """Устанавливает robust‑соединение и настраивает queue/exchange."""
        self.logger.debug("Connecting to RabbitMQ: %s", RABBIT_URL)
        self.connection = await aio_pika.connect_robust(RABBIT_URL)
        self.channel = await self.connection.channel()

        await self.channel.set_qos(
            prefetch_count=1,
            timeout=900,  # 15 минут
        )

        self.task_queue = await self.channel.declare_queue(
            TASK_QUEUE,
            durable=True,
        )

        self.cancel_queue = await self.channel.declare_queue(
            CANCEL_QUEUE,
            durable=True,
        )

        self.result_queue = await self.channel.declare_queue(
            RESULT_QUEUE,
            durable=True,
        )

        # Именованный direct exchange для статусов
        self.status_exchange = await self.channel.declare_exchange(
            STATUS_QUEUE,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        self.logger.info("Connected and queues declared")

    async def send_status(self, task_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
        """Публикация статуса выполнения задачи."""
        if not self.channel or not self.status_exchange:
            return

        payload: dict[str, Any] = {
            "task_id": task_id,
            "status": status,
        }
        if extra:
            payload.update(extra)

        body = json.dumps(payload).encode("utf-8")
        msg = aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self.status_exchange.publish(
            msg,
            routing_key=task_id,
        )
        self.logger.info("Status sent: %s %s", task_id, status)

    async def send_result_to_queue(self, task_id: str, result_s3_key: str) -> None:
        """Отправляет информацию о готовом результате в очередь обработки."""
        if not self.channel or not self.result_queue:
            return

        payload = {
            "task_id": task_id,
            "result_bucket": RESULT_BUCKET,
            "result_key": result_s3_key,
            "status": "completed",
        }

        msg = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self.channel.default_exchange.publish(
            msg,
            routing_key=RESULT_QUEUE,
        )
        self.logger.info("Result notification sent to queue: %s", result_s3_key)

    async def _handle_cancel_message(self, message: aio_pika.IncomingMessage) -> None:
        async with message.process():
            try:
                data = json.loads(message.body)
                task_id = data.get("task_id")
            except Exception:
                self.logger.exception("Bad cancel message: %r", message.body)
                return

            if task_id and task_id == self.current_task_id:
                self.logger.info("Cancel requested for task %s", task_id)
                self.cancel_event.set()
                await self.send_status(task_id, "cancel_requested")

    async def _consume_cancel(self) -> None:
        """Отдельный консьюмер для очереди отмен."""
        assert self.cancel_queue is not None
        await self.cancel_queue.consume(self._handle_cancel_message, no_ack=False)

    async def _process_task(self, message: aio_pika.IncomingMessage) -> None:
        async with message.process(ignore_processed=True):
            try:
                data = json.loads(message.body)
            except Exception:
                self.logger.exception("Failed to decode task message: %r", message.body)
                return

            task_id: str = data.get("task_id")
            s3_bucket = data.get("s3_bucket")
            s3_key = data.get("s3_key")
            language = data.get("language")
            min_speakers = data.get("min_speakers")
            max_speakers = data.get("max_speakers")
            target_sample_rate = data.get("target_sample_rate", 16000)
            noise_sample_sec = data.get("noise_sample_sec", 0.5)

            if not task_id or not s3_bucket or not s3_key:
                self.logger.error("Task message missing required fields: %s", data)
                return

            self.current_task_id = task_id
            self.cancel_event.clear()
            await self.send_status(task_id, "received")

            # Временный файл для скачанного аудио
            temp_dir = Path(tempfile.gettempdir()) / "audio_worker"
            temp_dir.mkdir(exist_ok=True)
            local_audio_path = temp_dir / f"{task_id}_{Path(s3_key).name}"

            try:
                # Скачиваем файл из S3
                await self.send_status(task_id, "downloading", {"s3_key": s3_key})
                await self.s3_handler.download_file(s3_bucket, s3_key, local_audio_path)
                self.logger.info("Downloaded audio from S3: %s", s3_key)

                await self.send_status(task_id, "processing", {"stage": "transcription"})

                # Обработка в пуле потоков
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_transcription_pipeline,
                    str(local_audio_path),
                    language,
                    min_speakers,
                    max_speakers,
                    target_sample_rate,
                    noise_sample_sec,
                    task_id,
                )

                # Сохраняем результат в S3
                result_key = f"results/{task_id}/transcription.json"
                await self.send_status(task_id, "uploading_result", {"result_key": result_key})
                await self.s3_handler.upload_json(RESULT_BUCKET, result_key, result)
                self.logger.info("Uploaded result to S3: %s", result_key)

                # Отправляем в очередь результатов
                await self.send_result_to_queue(task_id, result_key)

                await self.send_status(
                    task_id,
                    "finished",
                    {
                        "result_bucket": RESULT_BUCKET,
                        "result_key": result_key,
                        "segments": len(result.get("segments", [])),
                    },
                )
                self.logger.info("Task %s finished", task_id)

            except asyncio.CancelledError:
                await self.send_status(task_id, "cancelled")
                self.logger.warning("Task %s cancelled", task_id)
                raise

            except Exception as e:
                self.logger.exception("Task %s failed: %s", task_id, e)
                await self.send_status(task_id, "error", {"error": str(e)})

            finally:
                # Удаляем временный файл
                if local_audio_path.exists():
                    local_audio_path.unlink()
                self.current_task_id = None

    def _run_transcription_pipeline(
        self,
        audio_path: str,
        language: str | None,
        min_speakers: int | None,
        max_speakers: int | None,
        target_sample_rate: int,
        noise_sample_sec: float,
        task_id: str,
    ) -> dict[str, Any]:
        """Выполняется в ThreadPoolExecutor."""
        result = self.transcriptor(
            Path(audio_path),
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            target_sample_rate=target_sample_rate,
            noise_sample_sec=noise_sample_sec,
        )
        return result

    async def _consume_tasks(self) -> None:
        """Основной консьюмер задач."""
        assert self.task_queue is not None

        async with self.task_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self.current_task_future and not self.current_task_future.done():
                    self.logger.warning("Got message while task is running; waiting...")
                    await self.current_task_future

                self.current_task_future = asyncio.create_task(self._process_task(message))

                try:
                    await self.current_task_future
                except asyncio.CancelledError:
                    self.logger.info("Current task future cancelled")

    async def run(self) -> None:
        """Основной цикл воркера с авто‑реконнектом."""
        while True:
            try:
                await self.connect()
                await asyncio.gather(
                    self._consume_tasks(),
                    self._consume_cancel(),
                )
            except asyncio.CancelledError:
                self.logger.info("Worker cancelled, shutting down...")
                break
            except Exception:
                self.logger.exception("Worker crashed, reconnect in 5s")
                await asyncio.sleep(5)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    hf_token = os.getenv("HF_SECRET_TOKEN")
    transcriptor = SpeechTranscriptor(
        whisper_model_name="small",
        diarizer_model_name="pyannote/speaker-diarization-3.1",
        hf_token=hf_token,
        device="cuda",
    )

    worker = AudioWorker(transcriptor)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
