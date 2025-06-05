# Summary of the timing

| Epsilon | Method | Median (s) | Min (s) | Min Params | Max (s) | Max Params |
|---------|--------|------------|---------|------------|---------|------------|
| 0.01    | TD     | 0.1159     | 0.0684  | mass_1: 5.60e+06<br>mass_2: 7.33<br>spin: 0.491<br>p0: 5.73<br>e0: 0.141 | 1.3490  | mass_1: 1.25e+05<br>mass_2: 0.627<br>spin: 0.875<br>p0: 14.10<br>e0: 0.585 |
| 0.01    | FD     | 0.1310     | 0.0719  | mass_1: 5.60e+06<br>mass_2: 7.33<br>spin: 0.491<br>p0: 5.73<br>e0: 0.141 | 1.7522  | mass_1: 1.48e+06<br>mass_2: 56.64<br>spin: 0.934<br>p0: 12.98<br>e0: 0.127 |
| 1e-05   | TD     | 0.1356     | 0.0808  | mass_1: 6.93e+06<br>mass_2: 6.93<br>spin: 0.872<br>p0: 3.86<br>e0: 0.00051 | 0.2733  | mass_1: 1.13e+05<br>mass_2: 8.75<br>spin: 0.493<br>p0: 24.63<br>e0: 0.896 |
| 1e-05   | FD     | 0.1517     | 0.0692  | mass_1: 9.41e+06<br>mass_2: 10.63<br>spin: 0.629<br>p0: 5.04<br>e0: 0.429 | 1.8511  | mass_1: 2.61e+05<br>mass_2: 21.02<br>spin: 0.990<br>p0: 19.96<br>e0: 0.889 |

# Detailed Timing Breakdown

## Case 1 (output type: fd)

- **Trajectory generation:** 0.1192 s (1.5%)
- **Amplitude generation:** 0.0183 s (0.2%)
- **Waveform summation:** 7.6873 s (98.2%)
- **Total:** 7.8270 s (matches 7.8248 s)

## Case 2 (output type: td)

- **Trajectory generation:** 0.1167 s (14.4%)
- **Amplitude generation:** 0.0225 s (2.8%)
- **Waveform summation:** 0.6720 s (82.8%)
- **Total:** 0.8120 s (matches 0.8113 s)

Waveform summation dominates the computation time in both cases.