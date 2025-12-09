import asyncio
import json
import uuid
from pathlib import Path

import aio_pika
import aioboto3

# Конфигурация
RABBIT_URL = "amqp://admin:admin@0.0.0.0:5672/"
TASK_QUEUE = "audio_tasks"
STATUS_QUEUE = "audio_status"
CANCEL_QUEUE = "audio_cancel"

MINIO_ENDPOINT = "http://0.0.0.0:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "adminpassword"
MINIO_BUCKET = "audio-files"


async def upload_file_to_s3(file_path: str) -> str:
    """Загружает файл в MinIO и возвращает S3 ключ."""
    session = aioboto3.Session()
    
    async with session.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    ) as s3_client:
        # Создаем бакет если не существует
        try:
            await s3_client.head_bucket(Bucket=MINIO_BUCKET)
        except Exception:
            await s3_client.create_bucket(Bucket=MINIO_BUCKET)
        
        # Генерируем уникальный ключ для файла
        file_name = Path(file_path).name
        s3_key = f"{uuid.uuid4()}/{file_name}"
        
        # Загружаем файл
        with open(file_path, "rb") as f:
            await s3_client.put_object(
                Bucket=MINIO_BUCKET,
                Key=s3_key,
                Body=f,
            )
        
        return s3_key


async def send_task(audio_path: str) -> str:
    """Загружает аудио в S3 и отправляет задачу в очередь."""
    # Загружаем файл в S3
    s3_key = await upload_file_to_s3(audio_path)
    
    # Отправляем задачу в RabbitMQ
    conn = await aio_pika.connect_robust(RABBIT_URL)
    async with conn:
        channel = await conn.channel()
        await channel.declare_queue(TASK_QUEUE, durable=True)

        task_id = str(uuid.uuid4())
        payload = {
            "task_id": task_id,
            "s3_bucket": MINIO_BUCKET,
            "s3_key": s3_key,
            "language": "ru",
            "min_speakers": 1,
            "max_speakers": 3,
        }
        msg = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await channel.default_exchange.publish(msg, routing_key=TASK_QUEUE)
        print(f"Task {task_id} sent with S3 key: {s3_key}")
        return task_id


async def listen_status(task_id: str) -> None:
    """Слушает статусы задачи."""
    conn = await aio_pika.connect_robust(RABBIT_URL)
    async with conn:
        channel = await conn.channel()

        exchange = await channel.declare_exchange(
            STATUS_QUEUE,
            aio_pika.ExchangeType.DIRECT,
            durable=True,
        )

        queue = await channel.declare_queue(exclusive=True, auto_delete=True)
        await queue.bind(exchange, routing_key=task_id)

        async with queue.iterator() as q_iter:
            async for msg in q_iter:
                async with msg.process():
                    data = json.loads(msg.body)
                    print("STATUS:", data)
                    if data.get("status") in {"finished", "cancelled", "error"}:
                        break


async def cancel_task(task_id: str) -> None:
    """Отправляет запрос на отмену задачи."""
    conn = await aio_pika.connect_robust(RABBIT_URL)
    async with conn:
        channel = await conn.channel()
        await channel.declare_queue(CANCEL_QUEUE, durable=True)

        payload = {"task_id": task_id}
        msg = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )
        await channel.default_exchange.publish(msg, routing_key=CANCEL_QUEUE)
        print(f"Cancel request sent for task {task_id}")


async def main() -> None:
    task_id = await send_task("data/vazelin.ogg")

    status_task = asyncio.create_task(listen_status(task_id))
    await asyncio.sleep(5)
    # await cancel_task(task_id)  # Раскомментируй для теста отмены

    await status_task


if __name__ == "__main__":
    asyncio.run(main())
