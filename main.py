from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi import HTTPException
import shutil
import uuid
import soundfile as sf
import os

from denoise import denoise_audio, estimate_snr

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Audio Denoising API is running"}


@app.post("/denoise", response_class=FileResponse)
async def denoise(file: UploadFile = File(...)):

    input_filename = f"input_{uuid.uuid4()}.wav"
    output_filename = f"output_{uuid.uuid4()}.wav"

    try:
        # Save input file
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process audio
        cleaned_audio, sr = denoise_audio(input_filename)

        # Compute SNR (optional)
        snr_value = estimate_snr(cleaned_audio)

        print(f"SNR Estimate: {snr_value:.2f} dB")  # debug only

        # Save output
        sf.write(output_filename, cleaned_audio, sr)

        # Return file
        return FileResponse(
            output_filename,
            media_type="audio/wav",
            filename="cleaned.wav"
        )
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup (VERY IMPORTANT)
        if os.path.exists(input_filename):
            os.remove(input_filename)
            

