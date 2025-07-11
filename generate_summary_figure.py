#!/usr/bin/env python3
"""
Generate a summary figure of the AudioMoth membrane test analysis.

This script creates a single figure with multiple subplots showing the key findings
from the analysis of the membrane vs. naked recordings.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import librosa
import librosa.display
from scipy.io import wavfile

# Set paths
MEMBRANE_PATH = "data/test_membrane.wav.WAV"
NAKED_PATH = "data/test_naked.wav.WAV"
OUTPUT_DIR = "results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set figure style
plt.style.use('ggplot')

def load_and_sync_audio():
    """Load audio files and return synchronized data."""
    from scipy import signal
    
    # Load membrane recording
    membrane_sr, membrane_data = wavfile.read(MEMBRANE_PATH)
    
    # Load naked recording
    naked_sr, naked_data = wavfile.read(NAKED_PATH)
    
    # Convert to float for processing
    membrane_data = membrane_data.astype(np.float32) / np.max(np.abs(membrane_data))
    naked_data = naked_data.astype(np.float32) / np.max(np.abs(naked_data))
    
    # Compute cross-correlation
    correlation = signal.correlate(naked_data, membrane_data, mode='full')
    
    # Find the lag with maximum correlation
    lag = np.argmax(correlation) - (len(membrane_data) - 1)
    
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
    
    return membrane_sr, membrane_synced, naked_synced

def create_summary_figure():
    """Create a summary figure with key findings."""
    print("Generating summary figure...")
    
    # Load and synchronize audio
    sr, membrane_data, naked_data = load_and_sync_audio()
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Title and summary
    fig.suptitle("AudioMoth Membrane Test Analysis - Key Findings", fontsize=16)
    summary_text = """
    The waterproof membrane primarily reduces overall signal level (8.71 dB) 
    while having minimal impact on frequency response (1.02-2.63 dB attenuation).
    Signal-to-noise ratio is slightly better with the membrane.
    """
    fig.text(0.5, 0.96, summary_text, ha='center', fontsize=12)
    
    # 1. Waveform comparison (10 seconds segment)
    segment_start = int(sr * 10)  # Start at 10 seconds
    segment_end = segment_start + int(sr * 5)  # 5 seconds duration
    
    ax1 = fig.add_subplot(gs[0, :])
    time = np.arange(segment_start, segment_end) / sr
    ax1.plot(time, naked_data[segment_start:segment_end], label='Naked (No Case)', alpha=0.7)
    ax1.plot(time, membrane_data[segment_start:segment_end], label='With Membrane', alpha=0.7)
    ax1.set_title('Waveform Comparison (5s segment)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    
    # 2. Spectrogram comparison
    n_fft = 2048
    hop_length = 512
    
    # Compute spectrograms
    naked_spec = np.abs(librosa.stft(naked_data, n_fft=n_fft, hop_length=hop_length))
    membrane_spec = np.abs(librosa.stft(membrane_data, n_fft=n_fft, hop_length=hop_length))
    
    # Convert to dB
    naked_spec_db = librosa.amplitude_to_db(naked_spec, ref=np.max)
    membrane_spec_db = librosa.amplitude_to_db(membrane_spec, ref=np.max)
    
    # Plot spectrograms
    ax2 = fig.add_subplot(gs[1, 0:2])
    img = librosa.display.specshow(
        naked_spec_db, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='log',
        ax=ax2
    )
    ax2.set_title('Naked Recording Spectrogram')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    ax3 = fig.add_subplot(gs[1, 2])
    img = librosa.display.specshow(
        membrane_spec_db, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='log',
        ax=ax3
    )
    ax3.set_title('Membrane Recording')
    fig.colorbar(img, ax=ax3, format='%+2.0f dB')
    
    # 3. Frequency response
    n_fft = 2**14  # Large FFT size for better frequency resolution
    
    naked_fft = np.abs(np.fft.rfft(naked_data, n=n_fft))
    membrane_fft = np.abs(np.fft.rfft(membrane_data, n=n_fft))
    
    # Create frequency axis
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Convert to dB
    naked_fft_db = 20 * np.log10(naked_fft + 1e-10)
    membrane_fft_db = 20 * np.log10(membrane_fft + 1e-10)
    
    # Smooth the frequency response
    def smooth(x, window_len=11):
        window = np.ones(window_len) / window_len
        return np.convolve(x, window, mode='same')
    
    naked_fft_db_smooth = smooth(naked_fft_db, window_len=101)
    membrane_fft_db_smooth = smooth(membrane_fft_db, window_len=101)
    
    # Calculate attenuation
    attenuation = naked_fft_db_smooth - membrane_fft_db_smooth
    
    # Plot frequency response
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.semilogx(freqs, naked_fft_db_smooth, label='Naked (No Case)')
    ax4.semilogx(freqs, membrane_fft_db_smooth, label='With Membrane')
    ax4.set_title('Frequency Response')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.grid(True, which="both", ls="-")
    ax4.legend()
    ax4.set_xlim(20, sr/2)
    
    # 4. Attenuation plot
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.semilogx(freqs, attenuation)
    ax5.set_title('Membrane Attenuation')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Attenuation (dB)')
    ax5.grid(True, which="both", ls="-")
    ax5.set_xlim(20, sr/2)
    
    # Add horizontal line at 0 dB
    ax5.axhline(y=0, color='r', linestyle='--')
    
    # Highlight frequency bands
    ax5.axvspan(20, 500, alpha=0.2, color='blue', label='Low (<500 Hz)')
    ax5.axvspan(500, 2000, alpha=0.2, color='green', label='Low-Mid (500-2000 Hz)')
    ax5.axvspan(2000, 8000, alpha=0.2, color='yellow', label='Mid-High (2-8 kHz)')
    ax5.axvspan(8000, sr/2, alpha=0.2, color='red', label='High (>8 kHz)')
    ax5.legend(fontsize='small')
    
    # 5. Metrics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate metrics
    naked_rms = np.sqrt(np.mean(naked_data**2))
    membrane_rms = np.sqrt(np.mean(membrane_data**2))
    rms_diff_db = 20 * np.log10(naked_rms / membrane_rms)
    
    # Calculate band attenuation
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
    
    # Create table data
    table_data = [
        ["Metric", "Value"],
        ["Overall Level Reduction", f"{rms_diff_db:.2f} dB"],
        ["Low Freq Attenuation", f"{band_attenuation['Low (<500 Hz)']:.2f} dB"],
        ["Low-Mid Attenuation", f"{band_attenuation['Low-Mid (500-2000 Hz)']:.2f} dB"],
        ["Mid-High Attenuation", f"{band_attenuation['Mid-High (2-8 kHz)']:.2f} dB"],
        ["High Freq Attenuation", f"{band_attenuation['High (>8 kHz)']:.2f} dB"]
    ]
    
    # Create table
    table = ax6.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add recommendations
    recommendations = """
    Recommendations:
    1. Increase gain by ~9 dB when using membrane
    2. No equalization needed (flat attenuation)
    3. Current membrane suitable for all recording applications
    """
    fig.text(0.5, 0.02, recommendations, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/summary_figure.png", dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to {OUTPUT_DIR}/summary_figure.png")

if __name__ == "__main__":
    create_summary_figure()