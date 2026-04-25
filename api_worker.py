import os
import gc
import uuid
import warnings
import time
import datetime
from contextlib import asynccontextmanager
from typing import Optional

# --- ПАТЧ СУМІСНОСТІ NUMPY 2.0 ---
import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "float"):
    np.float = float

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
from google.cloud import storage
from google.cloud import aiplatform
import whisper
import librosa
import uvicorn

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
models = {"whisper": None, "pyannote": None}

# Vertex AI — зчитуємо з env, щоб не хардкодити
GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION      = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_MODEL      = os.getenv("VERTEX_MODEL_RESOURCE_NAME", "")   # projects/.../models/...
GCS_BUCKET        = os.getenv("GCS_BUCKET", "bucket_audiov1")
GCS_OUTPUT_PREFIX = os.getenv("GCS_OUTPUT_PREFIX", "gs://bucket_audiov1/batch_results/")


# ──────────────────────────────────────────────
# ЛОГУВАННЯ
# ──────────────────────────────────────────────

def log(icon: str, message: str):
    print(f"{icon} {message}", flush=True)


# ──────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ МОДЕЛЕЙ (один раз при старті)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("🚀", f"Ініціалізація AI-сервісу | Пристрій: {device.upper()}")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        log("🔑", "HF_TOKEN знайдено — використовується персональний ключ.")
    else:
        log("⚠️", "HF_TOKEN не знайдено! Pyannote може не завантажитись.")

    try:
        log("🗣️", "КРОК 1/2: Завантаження Pyannote (діаризація)...")
        try:
            models["pyannote"] = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        except TypeError:
            log("🔄", "Стара версія Pipeline — fallback до use_auth_token...")
            models["pyannote"] = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )

        if device == "cuda":
            models["pyannote"].to(torch.device("cuda"))
        log("✅", "Pyannote готовий!")

        log("📦", "КРОК 2/2: Завантаження Whisper Medium...")
        models["whisper"] = whisper.load_model("medium", device=device)
        log("✅", "Whisper готовий!")

        log("🟢", "СЕРВЕР ГОТОВИЙ. Endpoints: /predict | /rawPredict | /batchPredict | /get-upload-url")
        yield

    except Exception as e:
        log("💥", f"КРИТИЧНА ПОМИЛКА ПРИ СТАРТІ: {e}")
        raise
    finally:
        log("🛑", "Зупинка сервера — очищення пам'яті...")
        models.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


app = FastAPI(title="AI Transcription Service v2", lifespan=lifespan)


# ──────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ — CLOUD STORAGE
# ──────────────────────────────────────────────

def download_from_gcs(gcs_path: str) -> str:
    """
    Завантажує файл із Google Cloud Storage у /tmp із унікальним іменем.
    Приклад: gs://bucket_audiov1/meeting_4.mp3
    """
    log("☁️", f"Завантаження з GCS: {gcs_path}")
    try:
        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)

        ext = os.path.splitext(blob_name)[-1] or ".tmp"
        local_path = f"/tmp/{uuid.uuid4()}_{blob_name.split('/')[-1]}"

        blob.download_to_filename(local_path)
        log("✅", f"GCS файл збережено: {local_path}")
        return local_path

    except Exception as e:
        log("❌", f"Помилка завантаження з GCS: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file from Cloud Storage")


# ──────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ — VERTEX AI BATCH
# ──────────────────────────────────────────────

def create_batch_prediction_job(
    gcs_source_uri: str,
    language: str,
    num_speakers: Optional[int],
    job_display_name: Optional[str] = None,
) -> dict:
    """
    Запускає асинхронне Batch Prediction завдання у Vertex AI.

    Vertex AI сам заберає файл із GCS, обробить і покладе результат
    у GCS_OUTPUT_PREFIX. Наш сервер при цьому не навантажується.

    Повертає resource_name завдання — фронтенд може використати його
    для опитування статусу через /batchStatus/{job_id}.
    """
    if not GCP_PROJECT_ID or not VERTEX_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Vertex AI не налаштований: перевірте GCP_PROJECT_ID та VERTEX_MODEL_RESOURCE_NAME."
        )

    log("🏭", f"Запуск Batch Prediction | Джерело: {gcs_source_uri}")

    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

    model_parameters = {"language": language}
    if num_speakers:
        model_parameters["num_speakers"] = num_speakers

    display_name = job_display_name or f"transcription_batch_{uuid.uuid4().hex[:8]}"

    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name=display_name,
        model_name=VERTEX_MODEL,
        instances_format="jsonl",
        gcs_source=gcs_source_uri,
        gcs_destination_prefix=GCS_OUTPUT_PREFIX,
        model_parameters=model_parameters,
        machine_type="n1-standard-4",
    )

    log("✅", f"Batch Job створено: {batch_job.resource_name} | Стан: {batch_job.state.name}")

    return {
        "job_resource_name": batch_job.resource_name,
        "job_display_name": display_name,
        "state": batch_job.state.name,
        "output_prefix": GCS_OUTPUT_PREFIX,
    }


# ──────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ — PIPELINE (ОНЛАЙН)
# ──────────────────────────────────────────────

def extract_speaker_segments(diarization) -> list[tuple]:
    """
    Витягує (start, end, speaker) з об'єкту pyannote Annotation або DiarizeOutput.
    """
    def _from_annotation(obj):
        if hasattr(obj, "itertracks"):
            return [(t.start, t.end, spk) for t, _, spk in obj.itertracks(yield_label=True)]
        return None

    segments = _from_annotation(diarization)
    if segments is not None:
        return segments

    log("🔎", f"Нестандартний тип {type(diarization).__name__} — глибоке сканування...")
    for attr in dir(diarization):
        if attr.startswith("_"):
            continue
        try:
            inner = getattr(diarization, attr)
            result = _from_annotation(inner)
            if result is not None:
                log("✅", f"Знайдено сегменти в атрибуті '{attr}'")
                return result
        except Exception:
            continue

    log("❌", f"Не вдалося розпакувати об'єкт. Атрибути: {dir(diarization)}")
    return []


def merge_whisper_and_diarization(whisper_segments: list, speaker_segments: list) -> list[dict]:
    """
    Для кожного Whisper-сегменту знаходить спікера з найбільшим перекриттям у часі.
    """
    results = []
    for seg in whisper_segments:
        w_start, w_end = seg["start"], seg["end"]
        text = seg["text"].strip()

        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0

        for d_start, d_end, speaker in speaker_segments:
            overlap = max(0.0, min(w_end, d_end) - max(w_start, d_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        results.append({
            "start": round(w_start, 2),
            "end": round(w_end, 2),
            "speaker": best_speaker,
            "text": text,
        })
    return results


async def _run_pipeline(
    tmp_path: str,
    language: str,
    num_speakers: Optional[int],
) -> JSONResponse:
    """
    Повний онлайн-pipeline: діаризація → транскрипція → злиття.
    Викликається з /predict і /rawPredict.
    """
    start_time = time.time()

    log("🎵", "Декодування аудіо (librosa, 16kHz)...")
    waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
    audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    if device == "cuda":
        audio_tensor = audio_tensor.to(device)
    log("✅", f"Аудіо: {waveform.shape[0] / sr:.1f}с, {sr}Hz")

    log("👥", "Аналіз спікерів (Pyannote)...")
    if num_speakers:
        log("🔢", f"Очікувана кількість спікерів: {num_speakers}")

    try:
        diarization = models["pyannote"](
            tmp_path,
            **({"num_speakers": num_speakers} if num_speakers else {})
        )
    except Exception as e:
        log("⚠️", f"Читання по шляху не вдалось ({e}), використовую тензор...")
        kwargs = {"waveform": audio_tensor, "sample_rate": sr}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        diarization = models["pyannote"](kwargs)

    speaker_segments = extract_speaker_segments(diarization)
    unique_speakers = len(set(s[2] for s in speaker_segments))
    log("✅", f"Знайдено {len(speaker_segments)} фрагментів від {unique_speakers} спікерів.")

    log("📝", f"Транскрипція (Whisper, мова: '{language}')...")
    whisper_result = models["whisper"].transcribe(
        tmp_path,
        language=language,
        fp16=(device == "cuda"),
        verbose=False,
    )
    log("✅", f"Розпізнано {len(whisper_result['segments'])} сегментів.")

    log("🔗", "Злиття транскрипції та діаризації...")
    merged = merge_whisper_and_diarization(whisper_result["segments"], speaker_segments)

    overall_text = " ".join(seg["text"] for seg in merged)
    processing_time = round(time.time() - start_time, 2)
    log("🟢", f"Готово! Час обробки: {processing_time}с")

    return JSONResponse(content={
        "status": "success",
        "data": {
            "overall_text": overall_text,
            "segments": merged,
            "processing_time": processing_time,
        },
    })


# ──────────────────────────────────────────────
# ENDPOINTS — ОНЛАЙН ОБРОБКА
# ──────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    gcs_path: Optional[str] = Form(None),
    language: str = Form("uk"),
    num_speakers: Optional[int] = Form(None),
):
    """
    Онлайн-обробка. Повертає результат синхронно (чекає завершення).

    Варіанти:
      multipart: file=<binary>, language="uk"
      multipart: gcs_path="gs://bucket_audiov1/file.mp3", language="uk"
    """
    if not models["whisper"] or not models["pyannote"]:
        raise HTTPException(status_code=503, detail="AI моделі ще завантажуються.")

    if not file and not gcs_path:
        raise HTTPException(status_code=400, detail="Потрібен або 'file', або 'gcs_path'.")

    tmp_path = None
    try:
        if gcs_path and gcs_path.startswith("gs://"):
            log("☁️", f"Режим GCS | URI: {gcs_path}")
            tmp_path = download_from_gcs(gcs_path)
        else:
            log("📤", f"Режим прямого завантаження | Файл: {file.filename}")
            tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename or 'audio'}"
            with open(tmp_path, "wb") as f:
                f.write(await file.read())
            log("💾", f"Збережено: {tmp_path}")

        return await _run_pipeline(tmp_path, language, num_speakers)

    except HTTPException:
        raise
    except Exception as e:
        log("❌", f"Помилка під час обробки: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


@app.post("/rawPredict")
async def raw_predict(
    file: Optional[UploadFile] = File(None),
    gcs_path: Optional[str] = Form(None),
    language: str = Form("uk"),
    num_speakers: Optional[int] = Form(None),
):
    """
    Альтернативний endpoint — ідентична логіка /predict.
    Для Vertex AI routing або прямих raw-запитів без проксі.
    """
    return await predict(
        file=file,
        gcs_path=gcs_path,
        language=language,
        num_speakers=num_speakers,
    )


# ──────────────────────────────────────────────
# ENDPOINTS — BATCH (АСИНХРОННА ОБРОБКА)
# ──────────────────────────────────────────────

@app.post("/batchPredict")
async def batch_predict(
    gcs_path: str = Form(...),
    language: str = Form("uk"),
    num_speakers: Optional[int] = Form(None),
    job_display_name: Optional[str] = Form(None),
):
    """
    Асинхронна batch-обробка через Vertex AI.

    Файл має бути вже в GCS (gcs_path обов'язковий).
    Сервер НЕ чекає результату — повертає resource_name завдання одразу.
    Результат Vertex AI покладе сам у GCS_OUTPUT_PREFIX.

    Приклад відповіді:
    {
      "status": "accepted",
      "data": {
        "job_resource_name": "projects/.../batchPredictionJobs/123",
        "job_display_name": "transcription_batch_abc123",
        "state": "JOB_STATE_QUEUED",
        "output_prefix": "gs://bucket_audiov1/batch_results/"
      }
    }
    """
    if not gcs_path.startswith("gs://"):
        raise HTTPException(
            status_code=400,
            detail="Для batch-режиму потрібен gcs_path у форматі gs://bucket/file."
        )

    log("🏭", f"Запит на Batch Prediction | Файл: {gcs_path} | Мова: {language}")

    try:
        job_info = create_batch_prediction_job(
            gcs_source_uri=gcs_path,
            language=language,
            num_speakers=num_speakers,
            job_display_name=job_display_name,
        )
        return JSONResponse(content={"status": "accepted", "data": job_info})

    except HTTPException:
        raise
    except Exception as e:
        log("❌", f"Помилка запуску Batch Job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batchStatus/{job_id}")
async def batch_status(job_id: str):
    """
    Перевірка стану Batch Job за його числовим ID (останній сегмент resource_name).

    Приклад: GET /batchStatus/1234567890
    """
    if not GCP_PROJECT_ID:
        raise HTTPException(status_code=503, detail="GCP_PROJECT_ID не налаштований.")

    log("🔍", f"Перевірка стану Batch Job ID: {job_id}")

    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        resource_name = f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/batchPredictionJobs/{job_id}"
        job = aiplatform.BatchPredictionJob(resource_name)

        return JSONResponse(content={
            "status": "success",
            "data": {
                "job_id": job_id,
                "job_display_name": job.display_name,
                "state": job.state.name,
                "output_prefix": GCS_OUTPUT_PREFIX,
                "create_time": str(job.create_time),
            },
        })

    except Exception as e:
        log("❌", f"Помилка отримання стану Job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# ENDPOINT — PRESIGNED URL ДЛЯ ПРЯМОГО UPLOAD
# ──────────────────────────────────────────────
from google.auth.transport import requests as google_requests
import google.auth

@app.post("/get-upload-url")
async def get_upload_url(filename: str):
    """
    Генерує тимчасовий підписаний URL (Signed URL v4) за допомогою IAM Credentials API.
    """
    log("🔗", f"Генерація Signed URL для файлу: {filename}")

    try:
        # 1. Отримуємо облікові дані Cloud Run
        credentials, project = google.auth.default()

        # 2. Якщо це дефолтні облікові дані Compute Engine (Cloud Run),
        # нам потрібен email сервісного акаунта
        auth_request = google_requests.Request()
        credentials.refresh(auth_request)
        service_account_email = credentials.service_account_email

        if not service_account_email:
             raise ValueError("Не вдалося визначити Service Account Email")

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET)
        blob_name = f"uploads/{uuid.uuid4()}_{filename}"
        blob = bucket.blob(blob_name)

        # 3. Використовуємо IAM Credentials API для підпису
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            service_account_email=service_account_email,
            access_token=credentials.token # Передаємо токен для автентифікації запиту на підпис
        )

        gcs_path = f"gs://{GCS_BUCKET}/{blob_name}"
        log("✅", f"Signed URL згенеровано | GCS шлях: {gcs_path}")

        return JSONResponse(content={
            "status": "success",
            "data": {
                "upload_url": url,
                "gcs_path": gcs_path,
                "expires_in_minutes": 15,
            },
        })

    except Exception as e:
        log("❌", f"Помилка генерації Signed URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# HEALTHCHECK
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Перевірка стану. Використовується Docker healthcheck і Vertex AI."""
    whisper_ok = models["whisper"] is not None
    pyannote_ok = models["pyannote"] is not None
    return {
        "status": "ready" if (whisper_ok and pyannote_ok) else "loading",
        "device": device,
        "models": {
            "whisper": "loaded" if whisper_ok else "not_loaded",
            "pyannote": "loaded" if pyannote_ok else "not_loaded",
        },
        "vertex_ai": {
            "project": GCP_PROJECT_ID or "not_set",
            "location": GCP_LOCATION,
            "model": VERTEX_MODEL or "not_set",
            "gcs_bucket": GCS_BUCKET,
            "output_prefix": GCS_OUTPUT_PREFIX,
        },
    }


# ──────────────────────────────────────────────
# ТОЧКА ВХОДУ
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)