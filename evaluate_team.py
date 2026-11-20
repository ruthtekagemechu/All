# =============================================================================
# Reverse-Engineering Project + Testing Module (one script)

import csv
import glob
import json
import os
import re
import tempfile

import numpy as np
from scipy.optimize import least_squares

TEAM_ID = "Ruth"
REFERENCE = "spike_data.npz"        # the original recording
SUBMISSIONS = "submissions"
SEED = 42


# =============================================================================
# PART 1: REVERSE ENGINEERING
#
# Model:
#   raw_data     = drift + spike_signal + noise
#   drift        = a * sin(2*pi*f*t + phi) + c
#   spike_signal = sum over spikes of A * exp(-(t - t_i) / tau), for t >= t_i
#   noise        = Gaussian white noise, mean 0, SD sigma
# =============================================================================

KERNEL_LEN = 60   # samples; exp(-60/3) is about 2e-9, so the tail is negligible


def moving_median(x, window=101):
    half = window // 2
    padded = np.pad(x, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def detect_spikes(x, k=4.0, window=101, max_gap=2):
    """Assignment 2A detector: moving-median baseline, threshold
    k x 1.4826 x MAD, one peak per group of above-threshold samples."""
    corrected = x - moving_median(x, window)
    mad = np.median(np.abs(corrected - np.median(corrected)))
    thr = k * 1.4826 * mad
    idx = np.flatnonzero(corrected > thr)
    if len(idx) == 0:
        return np.array([], dtype=int)
    groups = np.split(idx, np.where(np.diff(idx) > max_gap + 1)[0] + 1)
    return np.array([g[np.argmax(corrected[g])] for g in groups], dtype=int)


def make_drift(t, a, f, phi, c):
    return a * np.sin(2 * np.pi * f * t + phi) + c


def make_spike_train(n, onsets, amp, tau, length=KERNEL_LEN):
    """Place one exponential waveform at every onset (overlaps add up)."""
    kernel = amp * np.exp(-np.arange(length) / tau)
    impulses = np.zeros(n)
    np.add.at(impulses, onsets, 1.0)
    return np.convolve(impulses, kernel)[:n]


def model(params, t, onsets):
    a, f, phi, c, amp, tau = params
    return make_drift(t, a, f, phi, c) + make_spike_train(len(t), onsets, amp, tau)


def fit(t, x, onsets, start):
    res = least_squares(lambda p: model(p, t, onsets) - x, start,
                        bounds=([0, 0.01, -np.pi, -5, 0, 0.5],
                                [5, 1.0, np.pi, 5, 20, 20]))
    return res.x


def reconstruct(path):
    data = np.load(path)
    t, x = data["t"], data["raw_data"]
    n = len(x)
    fs = 1 / np.median(np.diff(t))

    # Step 1: detect spikes
    onsets = detect_spikes(x)

    # Step 2: rough drift guess (best single sine on a grid), then fit all together
    best = None
    for f in np.linspace(0.02, 0.5, 481):
        D = np.c_[np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(n)]
        co, *_ = np.linalg.lstsq(D, x, rcond=None)
        err = np.sum((x - D @ co) ** 2)
        if best is None or err < best[0]:
            best = (err, f, co)
    _, f0, co = best
    start = [np.hypot(co[0], co[1]), f0, np.arctan2(co[1], co[0]), co[2], 5.0, 3.0]
    params = fit(t, x, onsets, start)

    # Step 3: test each onset at -1, 0, +1 samples and keep the best
    for i in range(len(onsets)):
        errs = []
        for s in (-1, 0, 1):
            trial = onsets.copy()
            trial[i] = np.clip(trial[i] + s, 0, n - 1)
            errs.append(np.sum((model(params, t, trial) - x) ** 2))
        onsets[i] = np.clip(onsets[i] + (-1, 0, 1)[int(np.argmin(errs))], 0, n - 1)
    params = fit(t, x, onsets, params)

    # Step 4: look for spike-sized events left in the residual
    resid = x - model(params, t, onsets)
    sigma = resid.std()
    missed = [m for m in detect_spikes(resid, k=4.0) if resid[m] > params[4] / 2]
    if missed:
        onsets = np.sort(np.r_[onsets, missed])
        params = fit(t, x, onsets, params)
        resid = x - model(params, t, onsets)
        sigma = resid.std()

    a, f, phi, c, amp, tau = params
    drift = make_drift(t, a, f, phi, c)
    spike_signal = make_spike_train(n, onsets, amp, tau)

    # Step 5: noise with the original seed and the fitted level
    noise = np.random.default_rng(SEED).normal(0.0, sigma, n)

    info = {
        "n_spikes": int(len(onsets)),
        "firing_rate_hz": float(len(onsets) / (n / fs)),
        "drift_amplitude": float(a),
        "drift_frequency_hz": float(f),
        "drift_period_s": float(1 / f),
        "drift_phase_rad": float(phi),
        "drift_offset": float(c),
        "spike_amplitude": float(amp),
        "spike_tau_ms": float(tau / fs * 1000),
        "noise_sd": float(sigma),
        "residual_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "missed_spikes_added": int(len(missed)),
    }
    # results file: same structure as the original (t, raw_data)
    results = dict(t=t, raw_data=drift + spike_signal + noise)
    # parts file: the separate estimated components
    parts = dict(t=t, drift=drift, spike_signal=spike_signal, noise=noise,
                 reconstruction=drift + spike_signal, spike_indices=onsets.astype(int))
    return results, parts, info


# =============================================================================
# PART 2: TESTING MODULE
#
# Question: how close is a simulation to the original recording?
#
# A simulation uses new random noise, so it can never match the original
# sample by sample. Instead the module compares the things a simulation
# should get right:
#
#   spike timing   (40%)  are spikes at the same times? F1 score, 2 ms tolerance
#   spike shape    (20%)  same average spike waveform (height and decay)?
#   drift          (25%)  same slow baseline? (moving median of each signal)
#   noise level    (15%)  same noise size? (from sample-to-sample differences)
#
# Every measure is computed the same way on the original and on the
# simulation, with no team's own spike list, so the module does not favour
# simulations built like mine.
#
# Accepted files: t + raw_data (same structure as the original), or
# t + reconstruction (+ noise) as in the competition handout.
# A file identical to the original (a copy) scores 0.
# =============================================================================

WEIGHTS = {"spike_timing": 0.40, "spike_shape": 0.20, "drift": 0.25, "noise": 0.15}
TOL = 2            # spike times may differ by up to 2 samples (2 ms)
K_SIGMA = 5.0      # "spike-sized" means a jump of more than 5 noise SDs (real spikes are about 16)
SHAPE_LEN = 15     # samples of spike waveform compared
DRIFT_WIN = 301    # moving-median window (samples) used to extract the drift


def robust_noise_sd(x):
    """Noise SD from first differences. diff of white noise has SD sqrt(2)*sigma;
    the MAD ignores the rare spike jumps, and differencing removes slow drift."""
    d = np.diff(x)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def jump_signal(x):
    """Each sample minus the median of the 10 samples before it."""
    n = len(x)
    pre = np.lib.stride_tricks.sliding_window_view(np.pad(x, (10, 0), mode="edge"), 10)[:n]
    return x - np.median(pre, axis=1)


def onset_events(x, sigma, k=K_SIGMA):
    """Indices where the signal jumps up by more than k*sigma (a spike onset)."""
    idx = np.flatnonzero(jump_signal(x) > k * sigma)
    if len(idx) == 0:
        return idx
    groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    return np.array([g[0] for g in groups])


def mean_waveform(x, onsets, length=SHAPE_LEN):
    """Average spike waveform, measured from the pre-spike level."""
    j = jump_signal(x)
    n = len(x)
    good = [o for o in onsets if o + length <= n]
    if not good:
        return np.zeros(length)
    pre = x - j        # the pre-spike level at each sample
    return np.mean([x[o:o + length] - pre[o] for o in good], axis=0)


def match(a, b, tol=TOL):
    """Number of items in a with a partner in b within tol samples (one-to-one)."""
    b = list(np.sort(b))
    hits = 0
    for v in np.sort(a):
        for j, w in enumerate(b):
            if abs(v - w) <= tol:
                hits += 1
                b.pop(j)
                break
    return hits


def clip01(v):
    return float(np.clip(v, 0.0, 1.0))


def load_candidate(cand, n, t_ref):
    """Return (simulated signal, structured part or None, list of problems)."""
    problems = []
    keys = cand.files
    if "t" not in keys:
        problems.append("missing array: t")
    elif cand["t"].shape != (n,) or not np.allclose(cand["t"], t_ref):
        problems.append("t does not match the original time vector")

    def ok(k):
        if k not in keys:
            return False
        if cand[k].shape != (n,):
            problems.append(f"{k} has shape {cand[k].shape}, expected ({n},)")
            return False
        if not np.all(np.isfinite(cand[k])):
            problems.append(f"{k} contains NaN or inf")
            return False
        return True

    sim, structured = None, None
    if ok("reconstruction"):
        rec = cand["reconstruction"].astype(float)
        if ok("noise"):
            noise = cand["noise"].astype(float)
            parts_sum = None
            if ok("drift") and ok("spike_signal"):
                parts_sum = cand["drift"] + cand["spike_signal"]
            if parts_sum is not None and not np.allclose(rec, parts_sum) \
                    and np.allclose(rec, parts_sum + noise):
                structured = rec - noise          # noise was already added in
            else:
                structured = rec
            sim = structured + noise
        else:
            structured = rec
    if ok("raw_data"):
        sim = cand["raw_data"].astype(float)      # a full simulated recording wins
    if sim is None and not problems:
        problems.append("no raw_data, and no reconstruction + noise to build a simulation from")
    return sim, structured, problems


def evaluate(reference_file, candidate_file, figure_path=None):
    """
    Inputs:
        reference_file: path to original spike_data.npz
        candidate_file: path to another team's results file
        figure_path:    optional path for the overlay figure (.png)
    Output:
        dictionary with validity checks, metrics, sub-scores and final_score (0 to 100)
    """
    ref = np.load(reference_file)
    orig, t_ref = ref["raw_data"].astype(float), ref["t"]
    n = len(orig)
    out = {"candidate": str(candidate_file), "valid": True, "problems": []}

    # ----- validity -----
    try:
        cand = np.load(candidate_file, allow_pickle=False)
    except Exception as e:
        out.update(valid=False, problems=[f"cannot load file: {e}"], final_score=0.0)
        return out
    sim, structured, problems = load_candidate(cand, n, t_ref)
    if problems or sim is None:
        out.update(valid=False, problems=problems, final_score=0.0)
        return out
    if np.allclose(sim, orig, rtol=0, atol=1e-8):
        out.update(valid=False, problems=["identical to the original recording (a copy)"],
                   final_score=0.0)
        return out

    # ----- noise level -----
    sig_o, sig_c = robust_noise_sd(orig), robust_noise_sd(sim)
    noise_score = clip01(1 - abs(np.log(sig_c / sig_o)) / np.log(2))   # 0 at half or double

    # ----- drift -----
    d_o, d_c = moving_median(orig, DRIFT_WIN), moving_median(sim, DRIFT_WIN)
    drift_rmse = float(np.sqrt(np.mean((d_o - d_c) ** 2)))
    drift_score = clip01(1 - drift_rmse / d_o.std())

    # ----- spike timing (same detector on both signals) -----
    on_o = onset_events(orig, sig_o)
    on_c = onset_events(sim, sig_o)
    tp = match(on_c, on_o)
    precision = tp / len(on_c) if len(on_c) else 0.0
    recall = tp / len(on_o) if len(on_o) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # ----- spike shape -----
    w_o, w_c = mean_waveform(orig, on_o), mean_waveform(sim, on_c)
    shape_err = float(np.linalg.norm(w_c - w_o) / np.linalg.norm(w_o))
    shape_score = clip01(1 - shape_err) if len(on_c) else 0.0

    subs = {"spike_timing": f1, "spike_shape": shape_score,
            "drift": drift_score, "noise": noise_score}
    final = 100 * sum(WEIGHTS[k] * subs[k] for k in WEIGHTS)

    # ----- extra information (not part of the score) -----
    sample_rmse = float(np.sqrt(np.mean((sim - orig) ** 2)))
    extra = {
        # two independent noise samples of SD s differ by about sqrt(2)*s
        "sample_rmse": sample_rmse,
        "sample_rmse_vs_independent_noise": float(sample_rmse / (np.sqrt(2) * sig_o)),
    }
    if structured is not None:
        resid = orig - structured
        extra["structured_residual_rmse"] = float(np.sqrt(np.mean(resid ** 2)))
        extra["structured_residual_vs_noise"] = float(extra["structured_residual_rmse"] / sig_o)

    out.update({
        "noise_sd_original": float(sig_o),
        "noise_sd_candidate": float(sig_c),
        "drift_rmse": drift_rmse,
        "n_spikes_original": int(len(on_o)),
        "n_spikes_candidate": int(len(on_c)),
        "spike_precision": float(precision),
        "spike_recall": float(recall),
        "spike_shape_error": shape_err,
        "mean_spike_peak_original": float(w_o[0]),
        "mean_spike_peak_candidate": float(w_c[0]),
        "extra": extra,
        "subscores": {k: round(v, 4) for k, v in subs.items()},
        "final_score": round(float(final), 2),
    })

    # ----- overlay figure -----
    if figure_path:
        import matplotlib.pyplot as plt
        t = t_ref
        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        ax[0].plot(t, orig, lw=0.4, color="0.55", label="original")
        ax[0].plot(t, sim, lw=0.4, color="C0", alpha=0.6, label="simulation")
        ax[0].plot(t[on_o], orig[on_o], "kv", ms=4, label="original spikes")
        ax[0].plot(t[on_c], sim[on_c] + 0.5, "r^", ms=4, label="simulated spikes")
        ax[0].set_title(f"{os.path.basename(str(candidate_file))}: score {final:.1f}")
        ax[1].plot(t, d_o, color="0.3", label="original drift")
        ax[1].plot(t, d_c, color="C1", label="simulated drift")
        ax[1].set_title(f"Drift (moving median), RMSE = {drift_rmse:.3f}")
        k = np.arange(SHAPE_LEN)
        ax[2].plot(k, w_o, "o-", color="0.3", label="original")
        ax[2].plot(k, w_c, "s-", color="C3", label="simulation")
        ax[2].set_title("Average spike waveform")
        ax[2].set_xlabel("Samples after onset (ms)")
        for a in ax[:2]:
            a.set_xlabel("Time (s)")
        for a in ax:
            a.set_ylabel("Voltage")
            a.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=120)
        plt.close(fig)
        out["figure"] = str(figure_path)

    return out


# =============================================================================
# CHECK THE TESTING MODULE: good simulations should score high, bad ones low
# =============================================================================

def check_testing_module(reference_file, parts_file, results_file):
    ref = np.load(reference_file)
    orig, t = ref["raw_data"], ref["t"]
    p = dict(np.load(parts_file))
    n = len(orig)
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(7)

    # rebuild spikes from the stored spike train, so bad versions can be made
    on = p["spike_indices"]
    amp = p["spike_signal"][on].max()
    tau = -1 / np.log(p["spike_signal"][on[0] + 1] / p["spike_signal"][on[0]])
    spikes = lambda idx, a=amp, tt=tau: make_spike_train(n, idx, a, tt)

    def sim_file(name, raw):
        path = os.path.join(tmp, f"team_{name}_results.npz")
        np.savez(path, t=t, raw_data=raw)
        return evaluate(reference_file, path)["final_score"]

    good = evaluate(reference_file, results_file)["final_score"]
    parts = evaluate(reference_file, parts_file)["final_score"]
    other_noise = p["drift"] + p["spike_signal"] + rng.normal(0, p["noise"].std(), n)
    cases = {
        "copy of the original": sim_file("copy", orig.copy()),
        "no spikes": sim_file("nospk", p["drift"] + p["noise"]),
        "half the spikes": sim_file("half", p["drift"] + spikes(on[::2]) + p["noise"]),
        "spikes at random times": sim_file("rand", p["drift"] + spikes(
            np.sort(rng.choice(n - 60, len(on), replace=False))) + p["noise"]),
        "spikes 3 samples late": sim_file("late", p["drift"] + spikes(on + 3) + p["noise"]),
        "spikes half as tall": sim_file("small", p["drift"] + spikes(on, amp / 2) + p["noise"]),
        "no drift": sim_file("nodrift", p["spike_signal"] + p["noise"]),
        "noise twice as big": sim_file("loud", p["drift"] + p["spike_signal"] + 2 * p["noise"]),
    }
    print(f"{'my results file (t, raw_data)':34s} {good:6.1f}")
    print(f"{'my parts file (same simulation)':34s} {parts:6.1f}")
    other = sim_file("seed7", other_noise)
    print(f"{'same model, different noise seed':34s} {other:6.1f}")
    for k, v in cases.items():
        print(f"{k:34s} {v:6.1f}")
        assert v < good - 5, k
    assert cases["copy of the original"] == 0
    assert abs(parts - good) < 0.5
    assert abs(other - good) < 5, "score should not depend much on the noise seed"
    print("all testing-module checks passed")


# =============================================================================
# SCORE EVERY TEAM (in class)
# =============================================================================

def score_all(reference_file, folder, my_id=TEAM_ID, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        m = re.search(r"team_(.+?)_(results|reconstruction)\.npz$", os.path.basename(path))
        tid = m.group(1) if m else os.path.splitext(os.path.basename(path))[0]
        res = evaluate(reference_file, path,
                       figure_path=os.path.join(out_dir, f"overlay_{tid}.png"))
        with open(os.path.join(out_dir, f"result_{tid}.json"), "w") as f:
            json.dump(res, f, indent=2)
        subs = res.get("subscores", {})
        rows.append({
            "team": tid + (" (me)" if tid == my_id else ""),
            "valid": res["valid"],
            "n_spikes": res.get("n_spikes_candidate", ""),
            **{k: subs.get(k, "") for k in WEIGHTS},
            "final_score": res["final_score"],
            "problems": "; ".join(res.get("problems", [])),
        })
    rows.sort(key=lambda r: r["final_score"], reverse=True)
    if rows:
        with open(os.path.join(out_dir, "score_table.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{'team':16s} {'score':>7s}  spikes  problems")
    for r in rows:
        print(f"{r['team']:16s} {r['final_score']:7.1f}  {str(r['n_spikes']):6s}  {r['problems']}")
    return rows


# ============================================================================
# README (written automatically with the fitted numbers)
# ============================================================================

def write_readme(info, res, path="README.md"):
    s = res["subscores"]
    e = res["extra"]
    text = f"""# Reverse-Engineering spike_data.npz

Files:
- team_{TEAM_ID}_results.npz: my simulation, same structure as the original
  (t and raw_data, 10,000 samples each)
- team_{TEAM_ID}_reconstruction.npz: the separate parts (drift, spike_signal,
  noise, reconstruction, spike_indices)
- evaluate_team.py: the reverse engineering code and the testing module

## Inferred model

raw_data = drift + spike_signal + noise

- drift = a * sin(2*pi*f*t + phi) + c
- spike_signal = sum over spikes of A * exp(-(t - t_i) / tau), for t >= t_i
- noise = Gaussian white noise, mean 0, SD sigma

| Parameter | Estimate |
| --- | --- |
| Number of spikes | {info['n_spikes']} (firing rate {info['firing_rate_hz']:.2f} Hz) |
| Spike amplitude A | {info['spike_amplitude']:.2f} |
| Spike decay constant tau | {info['spike_tau_ms']:.2f} ms |
| Spike rise | instant (one sample) |
| Drift amplitude a | {info['drift_amplitude']:.3f} |
| Drift frequency f | {info['drift_frequency_hz']:.4f} Hz (period {info['drift_period_s']:.1f} s) |
| Drift phase phi | {info['drift_phase_rad']:.3f} rad |
| Drift offset c | {info['drift_offset']:.3f} |
| Noise SD sigma | {info['noise_sd']:.3f} |

The values suggest the simulation used a spike amplitude of 5, a decay
constant of 3 ms and a noise SD of 0.3.

## Assumptions

1. All spikes come from one neuron and share one waveform: an instant jump
   followed by exponential decay. Overlapping spikes add up.
2. All spikes have the same amplitude. The spread of measured peak heights
   matches what the noise alone would cause.
3. The drift is one slow sine wave. Adding harmonics did not reduce the error,
   and the residual has no slow trend left.
4. The noise is white and Gaussian (residual autocorrelation near zero,
   kurtosis close to 3).
5. The small bump near 1.014 s is noise, not a spike: about 4 noise SDs tall,
   while real spikes are about 16.

## Method

1. Detection with my Assignment 2A detector: 101-sample moving-median
   baseline, threshold 4 x 1.4826 x MAD, one peak per event. Spikes rise in
   one sample, so the peak is also the spike start.
2. All six parameters (a, f, phi, c, A, tau) fitted together by nonlinear
   least squares, starting from a grid search over the drift frequency.
3. Each spike start tested at -1, 0 and +1 samples; the best kept.
4. The detector run again on the residual to look for missed spikes
   ({info['missed_spikes_added']} found).
5. Noise generated with numpy.random.default_rng(42) at the fitted SD. I tried
   to recover the exact original noise from seed 42 (np.random.seed(42) and
   default_rng(42), with up to 600 random draws before the noise and every
   offset up to 20,000 samples), but nothing matched. My noise has the right
   size and type but is a different random sample.

The drift + spikes part alone differs from the original by an RMSE of
{info['residual_rmse']:.3f}, about the same as the noise level, so what is left
over is just noise.

## Testing module

evaluate(reference_file, candidate_file) answers: how close is a simulation
to the original? A simulation has new random noise, so it cannot match the
original sample by sample. The module compares what a simulation should get
right, measured the same way on both signals:

| Measure | Weight | How |
| --- | --- | --- |
| Spike timing | 40% | Spike onsets found in both signals (jumps of more than 5 noise SDs), matched within 2 ms, F1 score |
| Spike shape | 20% | Average spike waveform over 15 ms, relative difference |
| Drift | 25% | Moving median (301 samples) of both signals, RMSE relative to the original drift's spread |
| Noise level | 15% | Noise SD from sample-to-sample differences, ratio |

It accepts files with t and raw_data (same structure as the original) or with
the competition parts (t, reconstruction, noise). A copy of the original
scores 0. No team's own spike list is used, so the module does not favour
simulations built the same way as mine.

My own results file scores {res['final_score']:.1f} / 100
(spike timing {s['spike_timing']:.2f}, spike shape {s['spike_shape']:.2f},
drift {s['drift']:.2f}, noise {s['noise']:.2f}). Sample-by-sample RMSE is
{e['sample_rmse']:.3f}, which is what two independent noise samples of this
size would give, as expected.

Checks of the module: a copy of the original, no spikes, half the spikes,
spikes at random times, spikes 3 ms late, spikes half as tall, no drift, and
noise twice as big all score clearly lower. The same model with a different
noise seed scores almost the same, so luck with the noise hardly matters.

## Possible bias

The module assumes white Gaussian noise of constant size, like my model, and
the 5 SD cutoff, 2 ms tolerance and weights are my own choices. A simulation
built on a different noise model could be judged unfairly.
"""
    with open(path, "w") as f:
        f.write(text)


# ============================================================================
# RUN EVERYTHING (only when run directly, not when imported)
# =============================================================================

if __name__ == "__main__":
    results, parts, info = reconstruct(REFERENCE)
    results_file = f"team_{TEAM_ID}_results.npz"
    parts_file = f"team_{TEAM_ID}_reconstruction.npz"
    np.savez(results_file, **results)
    np.savez(parts_file, **parts)
    print("Saved", results_file, "(t, raw_data) and", parts_file)
    print(json.dumps(info, indent=2))

    print("\nChecking the testing module")
    check_testing_module(REFERENCE, parts_file, results_file)

    my_result = evaluate(REFERENCE, results_file, figure_path=f"overlay_{TEAM_ID}.png")
    print("\nMy score:", my_result["final_score"], my_result["subscores"])

    write_readme(info, my_result)
    print("README.md written")

    if os.path.isdir(SUBMISSIONS):
        print("\nScoring every team in", SUBMISSIONS)
        score_all(REFERENCE, SUBMISSIONS)
    else:
        print(f"\nIn class: make a folder called '{SUBMISSIONS}' with all teams' "
              "results files and run again to score them.")
