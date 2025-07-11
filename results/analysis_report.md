# AudioMoth Membrane Test Analysis Report
*Generated on 2025-07-10 07:37:24*

## Overview
This report compares two AudioMoth recordings to evaluate the effect of a waterproof membrane on sound quality.

## Audio Metrics

| Metric | Naked (No Case) | With Membrane | Difference |
|--------|----------------|--------------|------------|
| RMS Level | 0.452698 | 0.166078 | 8.71 dB |
| Peak Level | 1.000000 | 1.000000 | 0.00 dB |
| Noise Floor | 0.000183 | 0.000153 | 1.58 dB |
| SNR | 74.75 dB | 76.33 dB | -1.58 dB |

## Frequency Band Attenuation

| Frequency Band | Average Attenuation |
|----------------|---------------------|
| Low (<500 Hz) | 2.63 dB |
| Low-Mid (500-2000 Hz) | 1.87 dB |
| Mid-High (2-8 kHz) | 1.02 dB |
| High (>8 kHz) | 2.26 dB |

## Interpretation

- **Overall Level**: The membrane reduces the overall signal level by 8.71 dB.
- **Signal-to-Noise Ratio**: The membrane has minimal effect on SNR (-1.58 dB difference).
- **Frequency Response**:
  - Low (<500 Hz): Minimal effect (2.63 dB)
  - Low-Mid (500-2000 Hz): Minimal effect (1.87 dB)
  - Mid-High (2-8 kHz): Minimal effect (1.02 dB)
  - High (>8 kHz): Minimal effect (2.26 dB)

## Conclusion

The waterproof membrane has a **minimal impact** on audio quality, with only slight attenuation across frequency bands. The most affected frequency range is Low (<500 Hz).

## Recommendations

- The current membrane is suitable for most recording applications with minimal impact on audio quality.

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
