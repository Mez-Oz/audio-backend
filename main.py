from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import uuid
import os

from pipeline import denoise_audio   # <-- IMPORTANT (updated import)

app = FastAPI()

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Audio Denoising API is running"}

@app.post("/denoise", response_class=FileResponse)
async def denoise(file: UploadFile = File(...)):

    # Keep original extension (important for mp3/webm support)
    ext = file.filename.split(".")[-1]

    input_filename = f"input_{uuid.uuid4()}.{ext}"
    output_filename = f"output_{uuid.uuid4()}.wav"

    try:
        # Save uploaded file
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run pipeline (this now SAVES file itself)
        denoise_audio(input_filename, output_filename)

        # Return cleaned file
        return FileResponse(
            output_filename,
            media_type="audio/wav",
            filename="cleaned.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup input file
        if os.path.exists(input_filename):
            os.remove(input_filename)

        # Optional: cleanup output after response (skip for now)
