#!/usr/bin/env python3
"""
AudioMoth Membrane Test Analysis

This script analyzes two AudioMoth recordings to compare the effect of a waterproof membrane
on sound quality. It synchronizes the recordings, performs various analyses, and generates
visualizations and a report.

Required libraries:
- librosa
- numpy
- scipy
- matplotlib
- soundfile
- pandas (optional, for report)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from datetime import datetime

# Set paths to audio files
MEMBRANE_PATH = "data/test_membrane.wav.WAV"
NAKED_PATH = "data/test_naked.wav.WAV"
OUTPUT_DIR = "results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set figure size and style for all plots
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use('ggplot')

def load_audio_files(membrane_path, naked_path):
    """
    Load the audio files and return their sample rates and data.
    
    Args:
        membrane_path: Path to the membrane recording
        naked_path: Path to the naked recording
        
    Returns:
        Tuple of (membrane_sr, membrane_data, naked_sr, naked_data)
    """
    print("Loading audio files...")
    
    # Load membrane recording
    membrane_sr, membrane_data = wavfile.read(membrane_path)
    
    # Load naked recording
    naked_sr, naked_data = wavfile.read(naked_path)
    
    # Convert to float for processing
    membrane_data = membrane_data.astype(np.float32) / np.max(np.abs(membrane_data))
    naked_data = naked_data.astype(np.float32) / np.max(np.abs(naked_data))
    
    print(f"Membrane recording: {len(membrane_data)/membrane_sr:.2f} seconds, {membrane_sr} Hz")
    print(f"Naked recording: {len(naked_data)/naked_sr:.2f} seconds, {naked_sr} Hz")
    
    # Ensure both files have the same sample rate
    if membrane_sr != naked_sr:
        print(f"Warning: Sample rates differ ({membrane_sr} Hz vs {naked_sr} Hz)")
        
    return membrane_sr, membrane_data, naked_sr, naked_data

def synchronize_recordings(membrane_data, naked_data, sr, plot=True):
    """
    Synchronize the two recordings using cross-correlation.
    
    Args:
        membrane_data: Audio data from membrane recording
        naked_data: Audio data from naked recording
        sr: Sample rate
        plot: Whether to plot the correlation
        
    Returns:
        Tuple of synchronized (membrane_data, naked_data)
    """
    print("Synchronizing recordings...")
    
    # Compute cross-correlation
    correlation = signal.correlate(naked_data, membrane_data, mode='full')
    
    # Find the lag with maximum correlation
    lag = np.argmax(correlation) - (len(membrane_data) - 1)
    print(f"Lag: {lag} samples ({lag/sr:.4f} seconds)")
    
    # Plot correlation if requested
    if plot:
        lags = np.arange(-len(membrane_data) + 1, len(naked_data))
        plt.figure()
        plt.plot(lags/sr, correlation)
        plt.axvline(x=lag/sr, color='r', linestyle='--')
        plt.title('Cross-correlation between recordings')
        plt.xlabel('Lag (seconds)')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cross_correlation.png", dpi=300)
    
    # Synchronize based on lag
    if lag > 0:
        # Membrane recording starts later
        membrane_synced = membrane_data
        naked_synced = naked_data[lag:]
        # Trim to same length
        min_len = min(len(membrane_synced), len(naked_synced))
        membrane_synced = membrane_synced[:min_len]
        naked_synced = naked_synced[:min_len]
    else:
        # Naked recording starts later
        membrane_synced = membrane_data[-lag:]
        naked_synced = naked_data
        # Trim to same length
        min_len = min(len(membrane_synced), len(naked_synced))
        membrane_synced = membrane_synced[:min_len]
        naked_synced = naked_synced[:min_len]
    
    print(f"Synchronized recordings: {min_len/sr:.2f} seconds")
    
    return membrane_synced, naked_synced

def plot_waveforms(membrane_data, naked_data, sr):
    """
    Plot the waveforms of both recordings.
    
    Args:
        membrane_data: Audio data from membrane recording
        naked_data: Audio data from naked recording
        sr: Sample rate
    """
    print("Plotting waveforms...")
    
    # Create time axis
    time = np.arange(0, len(membrane_data)) / sr
    
    # Plot waveforms
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, naked_data, label='Naked (No Case)')
    plt.title('Naked Recording Waveform')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, membrane_data, label='With Membrane', color='orange')
    plt.title('Membrane Recording Waveform')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot difference
    plt.subplot(3, 1, 3)
    plt.plot(time, naked_data - membrane_data, label='Difference', color='green')
    plt.title('Difference (Naked - Membrane)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/waveform_comparison.png", dpi=300)
    
    # Plot energy envelope
    plt.figure()
    
    # Calculate energy envelopes (using RMS with 100ms window)
    frame_length = int(sr * 0.1)  # 100ms window
    hop_length = int(frame_length / 2)  # 50% overlap
    
    naked_rms = librosa.feature.rms(y=naked_data, frame_length=frame_length, hop_length=hop_length)[0]
    membrane_rms = librosa.feature.rms(y=membrane_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Create time axis for RMS
    rms_time = np.arange(0, len(naked_rms)) * hop_length / sr
    
    plt.plot(rms_time, naked_rms, label='Naked (No Case)')
    plt.plot(rms_time, membrane_rms, label='With Membrane')
    plt.title('Energy Envelope Comparison')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/energy_comparison.png", dpi=300)

def plot_spectrograms(membrane_data, naked_data, sr):
    """
    Plot spectrograms of both recordings.
    
    Args:
        membrane_data: Audio data from membrane recording
        naked_data: Audio data from naked recording
        sr: Sample rate
    """
    print("Plotting spectrograms...")
    
    # STFT parameters
    n_fft = 2048
    hop_length = 512
    
    # Compute spectrograms
    naked_spec = np.abs(librosa.stft(naked_data, n_fft=n_fft, hop_length=hop_length))
    membrane_spec = np.abs(librosa.stft(membrane_data, n_fft=n_fft, hop_length=hop_length))
    
    # Convert to dB
    naked_spec_db = librosa.amplitude_to_db(naked_spec, ref=np.max)
    membrane_spec_db = librosa.amplitude_to_db(membrane_spec, ref=np.max)
    
    # Plot spectrograms
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(naked_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Naked Recording Spectrogram')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(membrane_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Membrane Recording Spectrogram')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectrogram_comparison.png", dpi=300)
    
    # Plot differential spectrogram
    plt.figure()
    diff_spec_db = naked_spec_db - membrane_spec_db
    librosa.display.specshow(diff_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Differential Spectrogram (Naked - Membrane)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/differential_spectrogram.png", dpi=300)
    
    # Plot log-frequency spectrograms for better visualization of higher frequencies
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(naked_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Naked Recording Spectrogram (Log Frequency)')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(membrane_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Membrane Recording Spectrogram (Log Frequency)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spectrogram_log_comparison.png", dpi=300)

def analyze_frequency_response(membrane_data, naked_data, sr):
    """
    Analyze and plot the frequency response of both recordings.
    
    Args:
        membrane_data: Audio data from membrane recording
        naked_data: Audio data from naked recording
        sr: Sample rate
    
    Returns:
        Dictionary with frequency analysis results
    """
    print("Analyzing frequency response...")
    
    # Compute FFT
    n_fft = 2**14  # Large FFT size for better frequency resolution
    
    naked_fft = np.abs(np.fft.rfft(naked_data, n=n_fft))
    membrane_fft = np.abs(np.fft.rfft(membrane_data, n=n_fft))
    
    # Create frequency axis
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Convert to dB
    naked_fft_db = 20 * np.log10(naked_fft + 1e-10)
    membrane_fft_db = 20 * np.log10(membrane_fft + 1e-10)
    
    # Smooth the frequency response for better visualization
    def smooth(x, window_len=11):
        window = np.ones(window_len) / window_len
        return np.convolve(x, window, mode='same')
    
    naked_fft_db_smooth = smooth(naked_fft_db, window_len=101)
    membrane_fft_db_smooth = smooth(membrane_fft_db, window_len=101)
    
    # Plot frequency response
    plt.figure()
    plt.semilogx(freqs, naked_fft_db_smooth, label='Naked (No Case)')
    plt.semilogx(freqs, membrane_fft_db_smooth, label='With Membrane')
    plt.title('Frequency Response Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.xlim(20, sr/2)  # Limit to audible range and Nyquist frequency
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/frequency_response.png", dpi=300)
    
    # Plot attenuation (difference in frequency response)
    plt.figure()
    attenuation = naked_fft_db_smooth - membrane_fft_db_smooth
    plt.semilogx(freqs, attenuation)
    plt.title('Membrane Attenuation (Naked - Membrane)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation (dB)')
    plt.grid(True, which="both", ls="-")
    plt.xlim(20, sr/2)
    
    # Add horizontal line at 0 dB
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Highlight frequency bands
    plt.axvspan(20, 500, alpha=0.2, color='blue', label='Low (<500 Hz)')
    plt.axvspan(500, 2000, alpha=0.2, color='green', label='Low-Mid (500-2000 Hz)')
    plt.axvspan(2000, 8000, alpha=0.2, color='yellow', label='Mid-High (2-8 kHz)')
    plt.axvspan(8000, sr/2, alpha=0.2, color='red', label='High (>8 kHz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attenuation.png", dpi=300)
    
    # Calculate average attenuation in different frequency bands
    bands = {
        "Low (<500 Hz)": (20, 500),
        "Low-Mid (500-2000 Hz)": (500, 2000),
        "Mid-High (2-8 kHz)": (2000, 8000),
        "High (>8 kHz)": (8000, sr/2)
    }
    
    band_attenuation = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Find indices corresponding to frequency range
        idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        # Calculate mean attenuation in this band
        band_attenuation[band_name] = np.mean(attenuation[idx])
    
    return {
        "frequency_bands": bands,
        "band_attenuation": band_attenuation,
        "freqs": freqs,
        "attenuation": attenuation
    }

def calculate_audio_metrics(membrane_data, naked_data):
    """
    Calculate various audio quality metrics.
    
    Args:
        membrane_data: Audio data from membrane recording
        naked_data: Audio data from naked recording
    
    Returns:
        Dictionary with calculated metrics
    """
    print("Calculating audio metrics...")
    
    # Calculate RMS levels
    naked_rms = np.sqrt(np.mean(naked_data**2))
    membrane_rms = np.sqrt(np.mean(membrane_data**2))
    
    # Calculate RMS difference in dB
    rms_diff_db = 20 * np.log10(naked_rms / membrane_rms)
    
    # Estimate noise floor (using the quietest 1% of samples)
    naked_noise = np.percentile(np.abs(naked_data), 1)
    membrane_noise = np.percentile(np.abs(membrane_data), 1)
    
    # Calculate signal peaks
    naked_peak = np.max(np.abs(naked_data))
    membrane_peak = np.max(np.abs(membrane_data))
    
    # Calculate SNR
    naked_snr = 20 * np.log10(naked_peak / naked_noise)
    membrane_snr = 20 * np.log10(membrane_peak / membrane_noise)
    
    # Calculate SNR difference
    snr_diff = naked_snr - membrane_snr
    
    return {
        "naked_rms": naked_rms,
        "membrane_rms": membrane_rms,
        "rms_diff_db": rms_diff_db,
        "naked_noise": naked_noise,
        "membrane_noise": membrane_noise,
        "naked_peak": naked_peak,
        "membrane_peak": membrane_peak,
        "naked_snr": naked_snr,
        "membrane_snr": membrane_snr,
        "snr_diff": snr_diff
    }

def generate_report(freq_analysis, audio_metrics):
    """
    Generate a markdown report with analysis results.
    
    Args:
        freq_analysis: Results from frequency analysis
        audio_metrics: Results from audio metrics calculation
    """
    print("Generating report...")
    
    # Create report content
    report = f"""# AudioMoth Membrane Test Analysis Report
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Overview
This report compares two AudioMoth recordings to evaluate the effect of a waterproof membrane on sound quality.

## Audio Metrics

| Metric | Naked (No Case) | With Membrane | Difference |
|--------|----------------|--------------|------------|
| RMS Level | {audio_metrics['naked_rms']:.6f} | {audio_metrics['membrane_rms']:.6f} | {audio_metrics['rms_diff_db']:.2f} dB |
| Peak Level | {audio_metrics['naked_peak']:.6f} | {audio_metrics['membrane_peak']:.6f} | {20 * np.log10(audio_metrics['naked_peak'] / audio_metrics['membrane_peak']):.2f} dB |
| Noise Floor | {audio_metrics['naked_noise']:.6f} | {audio_metrics['membrane_noise']:.6f} | {20 * np.log10(audio_metrics['naked_noise'] / audio_metrics['membrane_noise']):.2f} dB |
| SNR | {audio_metrics['naked_snr']:.2f} dB | {audio_metrics['membrane_snr']:.2f} dB | {audio_metrics['snr_diff']:.2f} dB |

## Frequency Band Attenuation

| Frequency Band | Average Attenuation |
|----------------|---------------------|
"""
    
    # Add frequency band attenuation data
    for band_name, attenuation in freq_analysis['band_attenuation'].items():
        report += f"| {band_name} | {attenuation:.2f} dB |\n"
    
    # Add interpretation
    report += """
## Interpretation

"""
    
    # Add interpretation based on the results
    if audio_metrics['rms_diff_db'] > 3:
        report += f"- **Overall Level**: The membrane reduces the overall signal level by {audio_metrics['rms_diff_db']:.2f} dB.\n"
    else:
        report += f"- **Overall Level**: The membrane has minimal effect on overall signal level ({audio_metrics['rms_diff_db']:.2f} dB difference).\n"
    
    if audio_metrics['snr_diff'] > 3:
        report += f"- **Signal-to-Noise Ratio**: The membrane reduces the SNR by {audio_metrics['snr_diff']:.2f} dB.\n"
    else:
        report += f"- **Signal-to-Noise Ratio**: The membrane has minimal effect on SNR ({audio_metrics['snr_diff']:.2f} dB difference).\n"
    
    # Add frequency band interpretation
    report += "- **Frequency Response**:\n"
    
    for band_name, attenuation in freq_analysis['band_attenuation'].items():
        if attenuation > 6:
            report += f"  - {band_name}: Significant attenuation ({attenuation:.2f} dB)\n"
        elif attenuation > 3:
            report += f"  - {band_name}: Moderate attenuation ({attenuation:.2f} dB)\n"
        elif attenuation < -3:
            report += f"  - {band_name}: Slight amplification ({attenuation:.2f} dB)\n"
        else:
            report += f"  - {band_name}: Minimal effect ({attenuation:.2f} dB)\n"
    
    # Add conclusion
    report += """
## Conclusion

"""
    
    # Determine overall impact
    max_attenuation = max(freq_analysis['band_attenuation'].values())
    if max_attenuation > 10:
        report += "The waterproof membrane has a **significant impact** on audio quality, with substantial attenuation in certain frequency bands. "
    elif max_attenuation > 5:
        report += "The waterproof membrane has a **moderate impact** on audio quality, with noticeable attenuation in certain frequency bands. "
    else:
        report += "The waterproof membrane has a **minimal impact** on audio quality, with only slight attenuation across frequency bands. "
    
    # Add most affected frequency range
    max_band = max(freq_analysis['band_attenuation'].items(), key=lambda x: x[1])[0]
    report += f"The most affected frequency range is {max_band}.\n"
    
    # Add recommendations
    report += """
## Recommendations

"""
    
    if max_attenuation > 10:
        report += "- Consider using a different membrane material with better acoustic properties.\n"
        report += "- If possible, apply equalization to recordings to compensate for the membrane's frequency response.\n"
    elif max_attenuation > 5:
        report += "- The current membrane is acceptable for general recording but may affect analysis of certain frequency bands.\n"
        report += "- Consider applying mild equalization to compensate for the membrane's frequency response.\n"
    else:
        report += "- The current membrane is suitable for most recording applications with minimal impact on audio quality.\n"
    
    # Add note about generated visualizations
    report += """
## Visualizations

The following visualizations have been generated and saved to the 'results' directory:

1. `waveform_comparison.png` - Waveform comparison of both recordings
2. `energy_comparison.png` - Energy envelope comparison
3. `spectrogram_comparison.png` - Spectrogram comparison
4. `spectrogram_log_comparison.png` - Log-frequency spectrogram comparison
5. `differential_spectrogram.png` - Differential spectrogram
6. `frequency_response.png` - Frequency response comparison
7. `attenuation.png` - Membrane attenuation across frequency spectrum
8. `cross_correlation.png` - Cross-correlation used for synchronization
"""
    
    # Write report to file
    with open(f"{OUTPUT_DIR}/analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"Report saved to {OUTPUT_DIR}/analysis_report.md")

def main():
    """Main function to run the analysis pipeline."""
    print("Starting AudioMoth membrane test analysis...")
    
    # Load audio files
    membrane_sr, membrane_data, naked_sr, naked_data = load_audio_files(MEMBRANE_PATH, NAKED_PATH)
    
    # Synchronize recordings
    membrane_synced, naked_synced = synchronize_recordings(membrane_data, naked_data, membrane_sr)
    
    # Plot waveforms
    plot_waveforms(membrane_synced, naked_synced, membrane_sr)
    
    # Plot spectrograms
    plot_spectrograms(membrane_synced, naked_synced, membrane_sr)
    
    # Analyze frequency response
    freq_analysis = analyze_frequency_response(membrane_synced, naked_synced, membrane_sr)
    
    # Calculate audio metrics
    audio_metrics = calculate_audio_metrics(membrane_synced, naked_synced)
    
    # Generate report
    generate_report(freq_analysis, audio_metrics)
    
    print("Analysis complete. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()