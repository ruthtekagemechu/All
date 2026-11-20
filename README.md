# Reverse-Engineering a Spike Recording

This project rebuilds a simulated neural recording (`spike_data.npz`) and includes a testing module that scores how close any simulation is to the original.

Everything is in one script, `evaluate_team.py`.

## Model

```
raw_data     = drift + spike_signal + noise
drift        = a * sin(2*pi*f*t + phi) + c
spike_signal = sum over spikes of A * exp(-(t - t_i) / tau), for t >= t_i
noise        = Gaussian white noise, mean 0, SD sigma
```

The recording has 42 spikes (4.20 Hz). My own simulation scores 97.6 out of 100 on the testing module.

## Part 1: Reverse engineering

1. Detect spikes with a moving-median baseline and a MAD threshold.
2. Fit all six parameters (a, f, phi, c, A, tau) together with nonlinear least squares, starting from a grid search over the drift frequency.
3. Test each spike start at -1, 0 and +1 samples and keep the best.
4. Run the detector again on the residual to catch missed spikes.
5. Generate new noise at the fitted SD.

## Part 2: Testing module

`evaluate(reference_file, candidate_file)` compares a simulation to the original on the things it should get right, measured the same way on both signals:

| Measure | Weight | How |
| --- | --- | --- |
| Spike timing | 40% | Spike onsets matched within 2 ms, F1 score |
| Spike shape | 20% | Average spike waveform over 15 ms, relative difference |
| Drift | 25% | Moving median of both signals, RMSE relative to the original drift's spread |
| Noise level | 15% | Noise SD from sample-to-sample differences, ratio |

A copy of the original scores 0. The script also checks itself: removing spikes, shifting them, shrinking them, removing drift or doubling the noise all score clearly lower.

## How to run

Put `spike_data.npz` in the same folder, then:

```
pip install -r requirements.txt
python evaluate_team.py
```

This saves:

- `team_Ruth_results.npz`: my simulation (`t` and `raw_data`)
- `team_Ruth_reconstruction.npz`: the separate parts (drift, spike_signal, noise, reconstruction, spike_indices)
- `overlay_Ruth.png`: original and simulation compared
- `README.md`: overwritten with the fitted parameter values

To score other teams, make a folder called `submissions` with their results files and run it again. Scores go to `results/score_table.csv`.

## Limitations

The testing module assumes white Gaussian noise of constant size, like my model. The 5 SD cutoff, 2 ms tolerance and weights are my own choices, so a simulation built on a different noise model could be judged unfairly.
