# denoise.py

import numpy as np
import librosa
from scipy.signal import butter, lfilter


# -------------------------------
# Bandpass Filter (Preprocessing)
# -------------------------------
def bandpass_filter(signal, sr, lowcut=80, highcut=7000, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, signal)

    return filtered


# --------------------------------------
# Wiener Filter (Decision Directed)
# --------------------------------------
def wiener_filter(y, sr, noise_est_dur=0.4, alpha_dd=0.98,
                  n_fft=2048, hop_length=512):

    # --- Noise estimation ---
    n_noise = int(noise_est_dur * sr)
    D_noise = librosa.stft(y[:n_noise], n_fft=n_fft, hop_length=hop_length)
    noise_psd = np.mean(np.abs(D_noise)**2, axis=1)

    # --- STFT of full signal ---
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    F, T = D.shape
    magnitude = np.abs(D)
    phase = np.exp(1j * np.angle(D))

    # --- Initialization ---
    gain = np.ones((F, T))
    prev_clean = magnitude[:, 0].copy()

    # --- Frame-by-frame processing ---
    for m in range(T):
        # Posteriori SNR
        snr_post = magnitude[:, m]**2 / (noise_psd + 1e-10) - 1
        snr_post = np.maximum(snr_post, 0)

        # A priori SNR
        snr_prior = alpha_dd * (prev_clean**2 / (noise_psd + 1e-10)) \
                    + (1 - alpha_dd) * snr_post

        # Wiener gain
        gain[:, m] = snr_prior / (1.0 + snr_prior)

        # Update previous clean estimate
        prev_clean = gain[:, m] * magnitude[:, m]

    # --- Reconstruction ---
    D_clean = gain * magnitude * phase
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))

    return y_clean


# --------------------------------------
# MAIN PIPELINE FUNCTION (IMPORTANT)
# --------------------------------------
def denoise_audio(file_path):
    """
    This is the ONLY function your backend will call.
    Input: file path
    Output: cleaned audio + sample rate
    """

    # Load audio
    audio, sr = librosa.load(file_path, sr=None)

    # Preprocess
    filtered_audio = bandpass_filter(audio, sr)

    # Denoise
    cleaned_audio = wiener_filter(filtered_audio, sr)

    # Normalize (IMPORTANT for audio quality)
    cleaned_audio = cleaned_audio / (np.max(np.abs(cleaned_audio)) + 1e-8)

    return cleaned_audio, sr

def estimate_snr(signal):
    power_signal = np.mean(signal**2)
    noise = signal - np.mean(signal)
    power_noise = np.mean(noise**2)
    return 10 * np.log10(power_signal / (power_noise + 1e-10))
