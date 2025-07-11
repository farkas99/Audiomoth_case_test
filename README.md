# AudioMoth Membrane Test Analysis

This project analyzes two AudioMoth recordings to compare the effect of a waterproof membrane on sound quality. It synchronizes the recordings, performs various analyses, and generates visualizations and reports.

## Overview

The analysis compares two recordings made with an AudioMoth acoustic logger:
1. **Membrane recording**: AudioMoth with waterproof membrane installed
2. **Naked recording**: AudioMoth without any case or membrane

The test audio was played from the YouTube video "Test Your Speakers/Headphone Sound Test" which includes various test segments (bass test, mid-range test, high-frequency test, etc.).

## Key Findings

- The membrane reduces the overall signal level by 8.71 dB
- Frequency response is minimally affected across all bands:
  - Low (<500 Hz): 2.63 dB attenuation
  - Low-Mid (500-2000 Hz): 1.87 dB attenuation
  - Mid-High (2-8 kHz): 1.02 dB attenuation
  - High (>8 kHz): 2.26 dB attenuation
- Signal-to-noise ratio is slightly better with the membrane (-1.58 dB difference)

The analysis reveals that the waterproof membrane primarily affects the overall amplitude of the sound but preserves the frequency characteristics quite well. The attenuation is relatively uniform across the frequency spectrum.

## Project Structure

```
.
├── data/                      # Input data directory
│   ├── CONFIG.TXT             # AudioMoth configuration file
│   ├── test_membrane.wav.WAV  # Recording with membrane
│   └── test_naked.wav.WAV     # Recording without membrane
├── results/                   # Output directory for analysis results
│   ├── analysis_report.md     # Basic analysis report
│   ├── enhanced_analysis_report.md  # Comprehensive analysis report
│   ├── summary_figure.png     # Summary figure with key findings
│   ├── waveform_comparison.png
│   ├── energy_comparison.png
│   ├── spectrogram_comparison.png
│   ├── spectrogram_log_comparison.png
│   ├── differential_spectrogram.png
│   ├── frequency_response.png
│   ├── attenuation.png
│   └── cross_correlation.png
├── audio_analysis.py          # Main analysis script
├── generate_summary_figure.py # Script to generate summary figure
└── README.md                  # This file
```

## Scripts

### audio_analysis.py

The main analysis script that performs the following steps:
1. Loads the audio files
2. Synchronizes the recordings using cross-correlation
3. Analyzes and compares waveforms, spectrograms, and frequency responses
4. Calculates audio metrics (RMS, SNR, etc.)
5. Generates visualizations and a report

### generate_summary_figure.py

Creates a single figure with multiple subplots showing the key findings from the analysis.

## Reports

Two reports are generated:

1. **analysis_report.md**: Basic analysis report with key metrics and findings
2. **enhanced_analysis_report.md**: Comprehensive report with detailed analysis, interpretations, and recommendations

## Visualizations

The analysis generates the following visualizations:

1. `waveform_comparison.png`: Waveform comparison of both recordings
2. `energy_comparison.png`: Energy envelope comparison
3. `spectrogram_comparison.png`: Spectrogram comparison
4. `spectrogram_log_comparison.png`: Log-frequency spectrogram comparison
5. `differential_spectrogram.png`: Differential spectrogram
6. `frequency_response.png`: Frequency response comparison
7. `attenuation.png`: Membrane attenuation across frequency spectrum
8. `cross_correlation.png`: Cross-correlation used for synchronization
9. `summary_figure.png`: Summary figure with key findings

## Recommendations

1. **Gain Compensation**: When using the membrane, increase the gain setting on the AudioMoth by approximately 9 dB to compensate for the level reduction.
2. **Minimal Processing Needed**: Since the frequency response is minimally affected, little to no equalization is needed when processing recordings made with the membrane.
3. **Suitable for Most Applications**: The current membrane is suitable for most recording applications where waterproofing is required, as it has minimal impact on audio quality beyond the overall level reduction.

## Requirements

- Python 3.6+
- Required libraries:
  - librosa
  - numpy
  - scipy
  - matplotlib
  - soundfile
  - pandas

## Usage

To run the main analysis:

```bash
python audio_analysis.py
```

To generate the summary figure:

```bash
python generate_summary_figure.py
```

## License

This project is open-source and available for educational and research purposes.