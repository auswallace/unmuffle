# Scorecard

Same script and voices in every scenario; only the noise changes. Plain Whisper = Whisper turbo on the mono mix.

| Scenario | WER plain | WER unmuffle | Words found plain | Words found unmuffle | Speakers right | SI-SDR raw | SI-SDR cleaned | DNSMOS raw | DNSMOS cleaned | Judge agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| pocket | 0.48 | 0.20 | 41% | 71% | 93% | -9.4 dB | -3.9 dB | 1.77 | 3.01 | 100% |
| cafe | 0.31 | 0.50 | 71% | 82% | 93% | 2.4 dB | 3.0 dB | 2.16 | 3.01 | 88% |
| wind | 0.06 | 0.07 | 95% | 92% | 100% | -0.7 dB | 9.7 dB | 2.58 | 3.44 | 93% |
| hum | 0.12 | 0.14 | 91% | 86% | 86% | 1.5 dB | 7.8 dB | 1.72 | 3.20 | 100% |
| traffic | 0.06 | 0.09 | 95% | 91% | 100% | -1.3 dB | 6.1 dB | 2.69 | 3.41 | 80% |
| music | 0.02 | 0.08 | 98% | 91% | 100% | 1.6 dB | 8.8 dB | 1.83 | 3.36 | 100% |
