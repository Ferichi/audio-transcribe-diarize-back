import os
import uuid
import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from google.cloud import storage, aiplatform
import uvicorn

# ──────────────────────────────────────────────
# КОНФІГУРАЦІЯ
# ──────────────────────────────────────────────

GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION      = os.getenv("GCP_LOCATION", "us-central1")
VERTEX_MODEL      = os.getenv("VERTEX_MODEL_RESOURCE_NAME", "")
GCS_BUCKET        = os.getenv("GCS_BUCKET", "bucket_audiov1")
GCS_OUTPUT_PREFIX = os.getenv("GCS_OUTPUT_PREFIX", "gs://bucket_audiov1/batch_results/")


# ──────────────────────────────────────────────
# ЛОГУВАННЯ
# ──────────────────────────────────────────────

def log(icon: str, message: str):
    print(f"{icon} {message}", flush=True)


app = FastAPI(title="Transcription Manager — Upload & Batch")


# ──────────────────────────────────────────────
# ENDPOINT 1 — PRESIGNED URL ДЛЯ ПРЯМОГО UPLOAD
# ──────────────────────────────────────────────

@app.post("/get-upload-url")
async def get_upload_url(filename: str):
    """
    Генерує тимчасовий Signed URL (v4, 15 хв) для прямого завантаження
    файлу з браузера у GCS — без проходження через цей сервер.

    Флоу на фронтенді:
      1. POST /get-upload-url?filename=meeting.mp3
         ← { upload_url, gcs_path, expires_in_minutes }
      2. PUT upload_url  (тіло — бінарний файл, Content-Type: application/octet-stream)
         ← файл потрапляє напряму в GCS
      3. POST /batch-predict  body: { gcs_path, language, num_speakers }
         ← job запущено асинхронно

    Потребує roles/iam.serviceAccountTokenCreator на сервісному акаунті.
    """
    log("🔗", f"Генерація Signed URL | Файл: {filename}")
    try:
        client = storage.Client()
        blob_name = f"uploads/{uuid.uuid4()}_{filename}"
        blob = client.bucket(GCS_BUCKET).blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="PUT",
            content_type="application/octet-stream",
        )

        gcs_path = f"gs://{GCS_BUCKET}/{blob_name}"
        log("✅", f"Signed URL готовий | GCS: {gcs_path}")

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
# ENDPOINT 2 — ЗАПУСК BATCH PREDICTION
# ──────────────────────────────────────────────

@app.post("/batch-predict")
async def batch_predict(
    gcs_path: str,
    language: str = "uk",
    num_speakers: int = None,
    job_display_name: str = None,
):
    """
    Запускає асинхронне Batch Prediction завдання у Vertex AI.

    Файл має бути вже в GCS (після /get-upload-url флоу).
    Сервер одразу повертає resource_name — результат Vertex AI
    покладе сам у GCS_OUTPUT_PREFIX коли завершить.
    """
    if not gcs_path.startswith("gs://"):
        raise HTTPException(
            status_code=400,
            detail="gcs_path має бути у форматі gs://bucket/file"
        )

    if not GCP_PROJECT_ID or not VERTEX_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Vertex AI не налаштований: перевірте GCP_PROJECT_ID та VERTEX_MODEL_RESOURCE_NAME."
        )

    display_name = job_display_name or f"transcription_batch_{uuid.uuid4().hex[:8]}"
    log("🏭", f"Запуск Batch Job | Файл: {gcs_path} | Мова: {language} | Job: {display_name}")

    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

        model_parameters = {"language": language}
        if num_speakers:
            model_parameters["num_speakers"] = num_speakers

        job = aiplatform.BatchPredictionJob.create(
            job_display_name=display_name,
            model_name=VERTEX_MODEL,
            instances_format="jsonl",
            gcs_source=gcs_path,
            gcs_destination_output_uri_prefix=GCS_OUTPUT_PREFIX,
            model_parameters=model_parameters,
            machine_type="n1-standard-4",
        )

        log("✅", f"Batch Job створено | Resource: {job.resource_name} | Стан: {job.state.name}")

        return JSONResponse(content={
            "status": "accepted",
            "data": {
                "job_resource_name": job.resource_name,
                "job_display_name": display_name,
                "state": job.state.name,
                "output_prefix": GCS_OUTPUT_PREFIX,
            },
        })

    except Exception as e:
        log("❌", f"Помилка запуску Batch Job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# ENDPOINT 3 — СТАТУС BATCH JOB
# ──────────────────────────────────────────────

@app.get("/batch-status/{job_id}")
async def batch_status(job_id: str):
    """
    Перевірка стану Batch Job за числовим ID
    (останній сегмент resource_name після останнього '/').

    Приклад: GET /batch-status/1234567890
    """
    if not GCP_PROJECT_ID:
        raise HTTPException(status_code=503, detail="GCP_PROJECT_ID не налаштований.")

    log("🔍", f"Перевірка стану Batch Job | ID: {job_id}")
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        resource_name = (
            f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
            f"/batchPredictionJobs/{job_id}"
        )
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
# HEALTHCHECK
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ready",
        "service": "transcription-manager",
        "config": {
            "gcs_bucket": GCS_BUCKET,
            "gcs_output_prefix": GCS_OUTPUT_PREFIX,
            "gcp_project": GCP_PROJECT_ID or "not_set",
            "gcp_location": GCP_LOCATION,
            "vertex_model": VERTEX_MODEL or "not_set",
        },
    }


# ──────────────────────────────────────────────
# ТОЧКА ВХОДУ
# ──────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)