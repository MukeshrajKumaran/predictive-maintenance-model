"""
GENERATE_DATA.PY
================
Simulates vibration sensor data from a rotating bearing.

In a real deployment, this data would come from an accelerometer
(ADXL345 over I2C, or MPU6050) mounted on the bearing housing,
sampled at 10-50 kHz and read by an STM32 or ESP32.

We simulate 4 conditions:
  0 - Normal       : clean sinusoidal signal + low noise
  1 - Inner Race   : fault frequency at BPFI (ball pass freq, inner race)
  2 - Outer Race   : fault frequency at BPFO
  3 - Ball Fault   : fault frequency at BSF (ball spin frequency)

These fault frequencies are real — they're calculated from bearing
geometry (pitch diameter, ball diameter, contact angle, RPM).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Bearing parameters (typical 6205 deep groove ball bearing)
SAMPLING_RATE = 12000   # Hz — typical for vibration acquisition
DURATION = 0.1          # seconds per sample
RPM = 1750
SHAFT_FREQ = RPM / 60   # ~29.2 Hz

# Characteristic defect frequencies (multiples of shaft frequency)
BPFI = 5.4152 * SHAFT_FREQ   # Ball Pass Frequency Inner race
BPFO = 3.5848 * SHAFT_FREQ   # Ball Pass Frequency Outer race
BSF  = 2.3573 * SHAFT_FREQ   # Ball Spin Frequency


def generate_signal(fault_type: int, n_samples: int) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SAMPLING_RATE * DURATION))

    # Base shaft rotation signal
    signal = np.sin(2 * np.pi * SHAFT_FREQ * t)

    if fault_type == 0:
        # Normal: clean signal + minimal noise
        noise_level = 0.05
    elif fault_type == 1:
        # Inner race fault: strong BPFI harmonic
        signal += 0.6 * np.sin(2 * np.pi * BPFI * t)
        signal += 0.3 * np.sin(2 * np.pi * 2 * BPFI * t)
        noise_level = 0.15
    elif fault_type == 2:
        # Outer race fault: strong BPFO harmonic
        signal += 0.7 * np.sin(2 * np.pi * BPFO * t)
        signal += 0.2 * np.sin(2 * np.pi * 2 * BPFO * t)
        noise_level = 0.15
    elif fault_type == 3:
        # Ball fault: BSF harmonic + modulation
        signal += 0.5 * np.sin(2 * np.pi * BSF * t)
        signal += 0.4 * np.sin(2 * np.pi * BSF * t) * np.sin(2 * np.pi * SHAFT_FREQ * t)
        noise_level = 0.2

    signal += np.random.normal(0, noise_level, len(t))
    return signal


def extract_features(signal: np.ndarray) -> dict:
    """
    Feature extraction — converts raw waveform into ML-ready numbers.

    TIME DOMAIN features capture amplitude characteristics:
      RMS    — overall vibration energy (most used in industry)
      Peak   — maximum amplitude
      Crest  — peak/RMS ratio, sensitive to impulses (spikes from faults)
      Kurtosis — statistical measure of impulsiveness, very sensitive to bearing faults
      Skewness — asymmetry of the signal distribution

    FREQUENCY DOMAIN features capture spectral content:
      We FFT the signal and look at energy in specific frequency bands.
      Fault frequencies show up as peaks in specific bands.
    """
    from scipy import stats
    from scipy.fft import fft, fftfreq

    n = len(signal)

    # --- Time domain ---
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-10)
    kurtosis = stats.kurtosis(signal)
    skewness = stats.skew(signal)
    variance = np.var(signal)
    mean_abs = np.mean(np.abs(signal))
    shape_factor = rms / (mean_abs + 1e-10)

    # --- Frequency domain ---
    freqs = fftfreq(n, 1 / SAMPLING_RATE)
    fft_magnitude = np.abs(fft(signal))[:n // 2]
    freqs = freqs[:n // 2]

    # Energy in bands around known fault frequencies (±10 Hz window)
    def band_energy(center_freq, bandwidth=10):
        mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
        return np.sum(fft_magnitude[mask] ** 2)

    bpfi_energy = band_energy(BPFI)
    bpfo_energy = band_energy(BPFO)
    bsf_energy  = band_energy(BSF)
    shaft_energy = band_energy(SHAFT_FREQ)

    # Spectral centroid — weighted average frequency
    spectral_centroid = np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-10)

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "variance": variance,
        "shape_factor": shape_factor,
        "bpfi_energy": bpfi_energy,
        "bpfo_energy": bpfo_energy,
        "bsf_energy": bsf_energy,
        "shaft_energy": shaft_energy,
        "spectral_centroid": spectral_centroid,
    }


def generate_dataset(samples_per_class=300):
    fault_names = {0: "Normal", 1: "Inner Race Fault", 2: "Outer Race Fault", 3: "Ball Fault"}
    records = []

    for fault_type in range(4):
        print(f"Generating {samples_per_class} samples for: {fault_names[fault_type]}")
        for _ in range(samples_per_class):
            signal = generate_signal(fault_type, samples_per_class)
            features = extract_features(signal)
            features["fault_type"] = fault_type
            features["fault_name"] = fault_names[fault_type]
            records.append(features)

    df = pd.DataFrame(records)
    df.to_csv("data/bearing_features.csv", index=False)
    print(f"\nDataset saved: {len(df)} samples, {len(df.columns)-2} features")
    return df


if __name__ == "__main__":
    generate_dataset()
