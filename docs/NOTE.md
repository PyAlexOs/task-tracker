Для поднятия микросервиса транскрибатора надо будет добавить:
```
sudo apt install ffmpeg nvidia-cuda-toolkit & sudo apt-get install libcudnn8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update && sudo apt upgrade
sudo apt install libcudnn8 libcudnn8-dev
```

---

Для присвоения имён спикерам и их распознавания в разных записях нужно использовать **speaker embeddings** — векторные представления голоса каждого спикера. Вот подход для решения вашей задачи

Для идентификации спикеров необходимо создать базу голосовых профилей (эмбеддингов) известных людей, а затем сравнивать сегменты из новых записей с этой базой через косинусное расстояние.

1. **Создание базы спикеров**: Для каждого известного человека запустите диаризацию на чистой записи его голоса и извлеките эмбеддинг через `pyannote/embedding` модель. Сохраните эти эмбеддинги в базу данных вместе с именами.[5][4]

2. **Извлечение эмбеддингов из сегментов**: В вашем методе `diarize()` используйте параметр `return_embeddings=True` при вызове pipeline:[4]
   ```python
   diarization, embeddings = self.diarization_pipeline(
       str(audio_path), 
       return_embeddings=True,
       **diarization_kwargs
   )
   ```

3. **Сопоставление через косинусное сходство**: Для каждого спикера в новой записи сравните его эмбеддинг с базой через `scipy.spatial.distance.cosine`. Если расстояние меньше порога (обычно 0.3-0.4), считайте спикера распознанным.[1][3]

## Пример кода для интеграции

```python
from pyannote.audio import Inference
from scipy.spatial.distance import cosine

# В __init__ добавьте:
embed_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
self.inference = Inference(embed_model, window="whole")

def recognize_speakers(self, diarization, embeddings, speaker_database, threshold=0.35):
    """Сопоставляет эмбеддинги с базой известных спикеров."""
    recognized = {}
    for idx, speaker_label in enumerate(diarization.labels()):
        embedding = embeddings[idx]
        distances = {name: cosine(embedding, db_emb) 
                    for name, db_emb in speaker_database.items()}
        min_speaker = min(distances, key=distances.get)
        if distances[min_speaker] < threshold:
            recognized[speaker_label] = min_speaker
        else:
            recognized[speaker_label] = f"Unknown_{speaker_label}"
    return recognized
```

База `speaker_database` должна быть словарём типа `{"Иван": embedding_array, "Мария": embedding_array}`, где эмбеддинги получены из эталонных записей каждого человека.[6][3]
