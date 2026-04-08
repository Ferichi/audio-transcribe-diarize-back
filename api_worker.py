import os
import gc
import warnings
import tempfile
import time
import sys
from contextlib import asynccontextmanager

# --- 1. ПАТЧ СУМІСНОСТІ NUMPY 2.0 ---
import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "inf"):
    np.inf = np.inf
if not hasattr(np, "float"):
    np.float = float

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
import librosa

# Приглушуємо попередження для чистоти консолі
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
models = {"whisper": None, "pyannote": None}


def log(icon, message):
    """Структурований вивід логів."""
    print(f"{icon} {message}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управління завантаженням моделей."""
    log("🚀", f"Ініціалізація AI-сервісу на пристрої: {device.upper()}")

    # Отримуємо токен безпечно
    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        log("🔑", "ТОКЕН ЗНАЙДЕНО: Система використовує твій персональний ключ.")

    try:
        from pyannote.audio import Pipeline
        import whisper

        log("🗣️", "КРОК 1: Завантаження Pyannote (діаризація)...")
        try:
            models["pyannote"] = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except TypeError:
            log("🔄", "Адаптація під стару версію Pipeline (use_auth_token)...")
            models["pyannote"] = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

        if device == "cuda":
            models["pyannote"].to(torch.device("cuda"))
        log("✅", "Pyannote готовий до роботи!")

        log("📦", "КРОК 2: Завантаження Whisper Medium (транскрибація)...")
        models["whisper"] = whisper.load_model("medium", device=device)
        log("✅", "Whisper готовий до роботи!")

        log("✨", "СЕРВЕР ПОВНІСТЮ ГОТОВИЙ ДО ТЕСТУВАННЯ!")
        yield
    except Exception as e:
        log("💥", f"КРИТИЧНА ПОМИЛКА ПРИ СТАРТІ: {e}")
        yield
    finally:
        log("🛑", "Зупинка сервера та очищення пам'яті...")
        models.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


app = FastAPI(title="AI Transcription Service", lifespan=lifespan)


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    if not models["whisper"] or not models["pyannote"]:
        raise HTTPException(status_code=503, detail="AI моделі ще завантажуються...")

    start_time = time.time()
    log("📥", f"Отримано файл для аналізу: {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        log("🎵", "Декодування аудіо (librosa)...")
        waveform, sr = librosa.load(tmp_path, sr=16000)
        audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        if device == "cuda":
            audio_tensor = audio_tensor.to(device)

        log("🧠", "Аналіз спікерів (Diarization)...")

        # СПРОБА 1: Пряме читання файлу. Часто обходить баг з форматом DiarizeOutput.
        try:
            diarization = models["pyannote"](tmp_path)
        except Exception as e:
            log("⚠️", f"Пряме читання файлу не вдалося ({e}), використовую тензор...")
            diarization = models["pyannote"]({"waveform": audio_tensor, "sample_rate": sr})

        speaker_segments = []

        # Внутрішня функція для перевірки, чи є об'єкт стандартною Анотацією
        def extract_segments(obj):
            if hasattr(obj, "itertracks"):
                return [(turn.start, turn.end, spk) for turn, _, spk in obj.itertracks(yield_label=True)]
            return None

        # МЕТОД 1: Стандартний
        segments = extract_segments(diarization)
        if segments is not None:
            speaker_segments = segments
        else:
            log("🔎", f"Об'єкт {type(diarization).__name__} нестандартний. Запускаю глибоке сканування...")
            # МЕТОД 2: Глибоке сканування (Рентген). Шукає Анотацію всередині DiarizeOutput.
            found = False
            for attr_name in dir(diarization):
                if not attr_name.startswith('_'):  # Ігноруємо системні змінні
                    try:
                        inner_obj = getattr(diarization, attr_name)
                        inner_segments = extract_segments(inner_obj)
                        if inner_segments is not None:
                            speaker_segments = inner_segments
                            log("✅", f"Знайдено приховані дані в атрибуті '{attr_name}'!")
                            found = True
                            break
                    except Exception:
                        continue

            # Якщо нічого не знайшли, виводимо структуру для діагностики
            if not found:
                log("❌", "Не вдалося розпакувати об'єкт. Структура:")
                log("❌", str(dir(diarization)))

        log("👥", f"Знайдено {len(speaker_segments)} фрагментів мовлення.")

        log("📝", "Розпізнавання тексту (Whisper)...")
        result = models["whisper"].transcribe(tmp_path, language="en", fp16=(device == "cuda"))

        log("🔗", "Об'єднання тексту з мітками спікерів...")
        final_results = []
        for seg in result["segments"]:
            w_s, w_e, text = seg["start"], seg["end"], seg["text"].strip()
            best_spk, max_ov = "Спікер ?", 0

            for d_s, d_e, spk in speaker_segments:
                ov = max(0, min(w_e, d_e) - max(w_s, d_s))
                if ov > max_ov:
                    max_ov, best_spk = ov, spk

            final_results.append({
                "start": round(w_s, 2),
                "end": round(w_e, 2),
                "speaker": best_spk,
                "text": text
            })

        proc_time = round(time.time() - start_time, 2)
        log("✨", f"Обробка завершена успішно за {proc_time}с!")
        return {"results": final_results, "duration": proc_time}

    except Exception as e:
        log("❌", f"Помилка під час обробки: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <title>AI Transcription Service</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f4f7f9; margin: 0; padding: 40px; }
            .card { background: white; max-width: 800px; margin: 0 auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            h1 { color: #1a73e8; text-align: center; }
            .upload-box { border: 2px dashed #1a73e8; padding: 30px; text-align: center; border-radius: 8px; background: #f8fbff; margin: 20px 0; cursor: pointer; }
            button { background: #1a73e8; color: white; border: none; padding: 15px; width: 100%; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; }
            #status { margin-top: 20px; text-align: center; color: #666; }
            .seg { margin-bottom: 15px; padding: 12px; background: #f9f9f9; border-left: 4px solid #1a73e8; border-radius: 4px; }
            .spk { font-weight: bold; color: #1a73e8; }
            .time { color: #888; font-size: 0.85em; margin-right: 10px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎙️ AI Transcription Service</h1>
            <div class="upload-box" onclick="document.getElementById('f').click()">
                <strong>📁 Оберіть аудіофайл для аналізу</strong>
                <input type="file" id="f" accept="audio/*" style="display: none;">
                <p id="fname" style="margin-top: 10px; color: #555;"></p>
            </div>
            <button onclick="run()">ПОЧАТИ ОБРОБКУ</button>
            <div id="status"></div>
            <div id="output"></div>
        </div>
        <script>
            document.getElementById('f').onchange = (e) => {
                document.getElementById('fname').innerText = e.target.files[0] ? e.target.files[0].name : '';
            };
            async function run(){
                const file = document.getElementById('f').files[0];
                if(!file) return alert("Оберіть файл!");
                const status = document.getElementById('status');
                const output = document.getElementById('output');
                status.innerText = "⏳ Аналіз триває... Слідкуйте за консоллю PyCharm.";
                output.innerHTML = "";
                const fd = new FormData(); fd.append('file', file);
                try {
                    const r = await fetch('/process-audio', {method:'POST', body:fd});
                    const d = await r.json();
                    if(d.results) {
                        status.innerText = `✅ Завершено за ${d.duration}с`;
                        d.results.forEach(i => {
                            output.innerHTML += `<div class="seg"><span class="time">[${i.start}с - ${i.end}с]</span><span class="spk">${i.speaker}:</span> ${i.text}</div>`;
                        });
                    } else { status.innerText = "❌ Помилка сервера."; }
                } catch(e) { status.innerText = "❌ Не вдалося зв'язатися з сервером."; }
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)