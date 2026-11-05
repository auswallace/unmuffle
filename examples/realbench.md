# Real-recording benchmark (MS-SNSD)

Real speech + real recorded noise at 0 dB SNR, 4 clips per file. Reference transcript = Whisper turbo on the clean speech.

| Noise | WER plain | WER unmuffle | SI-SDR raw | SI-SDR cleaned | DNSMOS raw | DNSMOS cleaned |
|---|---|---|---|---|---|---|
| airconditioner | 0.02 | 0.08 | -0.7 dB | 7.6 dB | 1.30 | 3.16 |
| airportannouncements | 0.20 | 0.21 | -0.7 dB | 8.3 dB | 1.20 | 3.11 |
| babble | 0.21 | 0.31 | -0.8 dB | 7.2 dB | 1.44 | 2.97 |
| copymachine | 0.30 | 0.22 | -0.7 dB | 8.0 dB | 1.14 | 3.02 |
| munching | 0.09 | 0.08 | -0.7 dB | 12.6 dB | 2.44 | 3.27 |
| neighbor | 0.10 | 0.17 | -0.8 dB | 7.5 dB | 2.03 | 3.11 |
| **mean** | 0.15 | 0.18 | -0.7 dB | 8.6 dB | 1.59 | 3.11 |
