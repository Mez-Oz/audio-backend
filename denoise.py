import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import torch
import torchaudio

# =============================
# CONFIG
# =============================

SR = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 512
LOWCUT = 100
HIGHCUT = 6000
ALPHA_DD = 0.98
BETA = 1.1
GAIN_FLOOR = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD MODEL (LOAD ONCE)
# =============================

bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
dl_model = bundle.get_model().to(DEVICE)
dl_model.eval()

# =============================
# DSP FUNCTIONS (UNCHANGED)
# =============================

def bandpass_filter(signal, lowcut, highcut, sr, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def pre_emphasis(signal):
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

def compute_stft(signal):
    return librosa.stft(signal, n_fft=FRAME_LENGTH,
                        hop_length=HOP_LENGTH)

def estimate_noise(signal):
    energy = librosa.feature.rms(y=signal)[0]
    threshold = np.percentile(energy, 20)

    stft = compute_stft(signal)
    mag = np.abs(stft)

    noise_frames = energy < threshold
    if np.sum(noise_frames) == 0:
        noise_frames[:5] = True

    noise_mag = mag[:, noise_frames]
    noise_psd = np.mean(noise_mag, axis=1)**2 + 1e-10

    return noise_psd

def wiener_filter(mag, noise_psd):
    F, T = mag.shape
    gain = np.ones((F, T))
    prev = mag[:, 0]

    for t in range(T):
        snr_post = np.maximum((mag[:, t]**2 / noise_psd) - 1, 0)
        snr_prior = ALPHA_DD * (prev**2 / noise_psd) + (1-ALPHA_DD)*snr_post
        gain[:, t] = snr_prior / (1 + snr_prior)
        prev = gain[:, t] * mag[:, t]

    return np.clip(gain, GAIN_FLOOR, 1)

def enhance(mag, gain, noise_psd):
    clean = np.maximum(mag * gain - BETA*np.sqrt(noise_psd[:, None]), 0)
    return np.maximum(clean, 0.02 * mag)

def reconstruct(mag, phase, length):
    return librosa.istft(mag * phase, hop_length=HOP_LENGTH, length=length)

# =============================
# DEMUCS (UNCHANGED)
# =============================

def deep_learning_denoise(signal):
    with torch.no_grad():
        x = torch.tensor(signal, dtype=torch.float32)

        x = x.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1).to(DEVICE)

        y = dl_model(x)

        y = y[0]
        y = y.mean(dim=0) * 0.8
        y = y.cpu().numpy()

    return y

# =============================
# MAIN PIPELINE
# =============================

def process_audio(signal):
    signal = bandpass_filter(signal, LOWCUT, HIGHCUT, SR)
    signal = pre_emphasis(signal)

    noise_psd = estimate_noise(signal)

    D = compute_stft(signal)
    mag, phase = np.abs(D), np.exp(1j * np.angle(D))

    gain = wiener_filter(mag, noise_psd)
    mag = enhance(mag, gain, noise_psd)

    output = reconstruct(mag, phase, len(signal))

    # Deep Learning refinement
    output = deep_learning_denoise(output)

    output = np.nan_to_num(output)
    output = np.clip(output, -1, 1).astype(np.float32)

    return output

# =============================
# BACKEND FUNCTION (IMPORTANT)
# =============================

def denoise_audio(input_path, output_path):
    """
    Backend entry point
    """

    # Load ANY format (wav/mp3/etc.)
    signal, sr = librosa.load(input_path, sr=SR)

    # Process
    cleaned = process_audio(signal)

    # Save as WAV
    sf.write(output_path, cleaned, sr)

    return output_path