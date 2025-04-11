from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import soundfile as sf
import pywt
from scipy.signal import lfilter
import os
import shutil
from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://sickcoder6184.github.io",
        "https://speech-compression.netlify.app"  # if you use Netlify too
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths
UPLOAD_FOLDER = "uploads/"
COMPRESSED_FOLDER = "compressed/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

# Convert FLAC to WAV
def convert_flac_to_wav(flac_path):
    wav_path = os.path.join(UPLOAD_FOLDER, os.path.basename(flac_path).replace(".flac", ".wav"))
    audio = AudioSegment.from_file(flac_path, format="flac")
    audio.export(wav_path, format="wav")
    return wav_path

# LPC
def apply_lpc(audio, order=10):
    a = librosa.lpc(audio, order=order)
    lpc_audio = lfilter([0] + -1 * a[1:], [1], audio)
    return lpc_audio

# DWT
def apply_dwt(audio):
    coeffs = pywt.wavedec(audio, 'db4', level=4)
    coeffs[1:] = [pywt.threshold(c, np.std(c) * 0.2, mode="soft") for c in coeffs[1:]]
    compressed_audio = pywt.waverec(coeffs, 'db4')
    return compressed_audio

# Full audio processing pipeline
def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    lpc_audio = apply_lpc(audio)
    compressed_audio = apply_dwt(lpc_audio)

    compressed_filename = os.path.basename(file_path).replace(".wav", "_compressed.flac")
    output_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)

    sf.write(output_path, compressed_audio, sr, format="FLAC")
    return output_path

# Upload route
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert FLAC to WAV
    if file_ext.lower() == "flac":
        file_path = convert_flac_to_wav(file_path)

    original_size = os.path.getsize(file_path)
    compressed_path = process_audio(file_path)
    compressed_size = os.path.getsize(compressed_path)
    file_url = f"http://127.0.0.1:8000/download/{os.path.basename(compressed_path)}"

    return {
        "message": "File processed",
        "compressed_file": file_url,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }

# Download route
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(COMPRESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/flac", filename=filename)
    return {"error": "File not found"}
