import logging
import os

import torch
from dotenv import load_dotenv

from src.speech_transcriptor.core.speech2text.local_transriptor import LocalSpeechRecognizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

recognizer = LocalSpeechRecognizer(
    model_name="base",
    hf_token=os.getenv("HF_SECRET_TOKEN"),
)

segments = recognizer.transcribe("vazelin.ogg", language="ru")
for seg in segments:
    logger.info(f"[{seg.start:.2f}s - {seg.end:.2f}s]: {seg.text}")


segments_with_speakers = recognizer.transcribe_with_diarization(
    "vazelin.ogg",
    language="ru",
    min_speakers=2,
    max_speakers=4,
)
for k, v in segments_with_speakers.items():
    print(k, v)
