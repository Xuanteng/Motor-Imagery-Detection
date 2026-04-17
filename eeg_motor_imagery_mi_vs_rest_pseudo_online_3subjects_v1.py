#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-online MI vs Rest detection on EEGBCI
Spyder-friendly single-file script
Batch version for 3 subjects

What this script does:
1) For each subject, build a within-subject train/test split
2) Train Riemannian + TangentSpace + LogisticRegression
3) Replay each test epoch with sliding windows
4) Save per-subject pseudo-online results
5) Save one combined summary JSON for subjects 1-3
    
April 14, 2026, by Xuanteng Yan
"""

from pathlib import Path
import json
import numpy as np
import mne
from mne.io import concatenate_raws

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# =========================
# User settings
# =========================
DATA_DIR = "/Users/xuantengyan/Library/CloudStorage/OneDrive-Personal/Others/Jobs/Ton Bridge Medical Device/EEGBCI_Data"
OUTPUT_DIR = "/Users/xuantengyan/Library/CloudStorage/OneDrive-Personal/Others/Jobs/Ton Bridge Medical Device/EEGBCI_Results"

SUBJECT_IDS = [1, 2, 3]
RUNS = [4, 8, 12]

L_FREQ = 8.0
H_FREQ = 30.0

# best window from your within-subject ablation
TMIN = 0.5
TMAX = 3.5

RANDOM_STATE = 42
TEST_SIZE = 0.2

# pseudo-online sliding window inside each epoch
WIN_SEC = 1.0
STEP_SEC = 0.25

# simple decision layer
SMOOTH_K = 3
THRESHOLD = 0.6
MIN_CONSECUTIVE_WINDOWS = 2


# =========================
# Helper functions
# =========================
def load_subject_paths(data_dir, subject_id, runs):
    return mne.datasets.eegbci.load_data(subject_id, runs, path=data_dir)



def load_subject_raw(paths):
    raws = []
    for path in paths:
        raw_part = mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel="auto",
            verbose="ERROR",
        )
        raws.append(raw_part)

    raw = concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)

    try:
        raw.set_montage("standard_1005")
    except Exception as e:
        print(f"Montage warning: {e}")

    return raw



def make_epochs_from_filtered_raw(raw, tmin, tmax):
    events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")
    print("Event map:", event_id_map)

    # EEGBCI:
    # T0 -> rest
    # T1 -> left
    # T2 -> right
    event_id = dict(rest=1, left=2, right=3)

    picks = mne.pick_types(
        raw.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude="bads",
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data().astype(np.float64)
    event_codes = epochs.events[:, 2].astype(np.int64)

    # rest = 0, MI(left/right) = 1
    y = np.where(event_codes == 1, 0, 1).astype(np.int64)

    return X, y, epochs.info["sfreq"]



def build_classifier():
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )



def moving_average(x, k=3):
    if k <= 1:
        return np.array(x, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - k + 1)
        out[i] = np.mean(x[start : i + 1])
    return out



def sliding_windows_from_epoch(X_epoch, sfreq, win_sec=1.0, step_sec=0.25):
    """
    X_epoch: (n_channels, n_times)
    Return:
        windows: (n_windows, n_channels, win_samples)
        times_epoch_sec: window center times, relative to epoch start
    """
    n_channels, n_times = X_epoch.shape
    win_samples = int(round(win_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))

    windows = []
    times_epoch_sec = []

    start = 0
    while start + win_samples <= n_times:
        end = start + win_samples
        windows.append(X_epoch[:, start:end])

        center_sample = start + win_samples / 2.0
        center_time = center_sample / sfreq
        times_epoch_sec.append(center_time)

        start += step_samples

    if len(windows) > 0:
        windows = np.stack(windows, axis=0)
    else:
        windows = np.empty((0, n_channels, win_samples))

    times_epoch_sec = np.array(times_epoch_sec, dtype=float)
    return windows, times_epoch_sec



def detect_from_probs(smoothed_probs, threshold=0.6, min_consecutive=2):
    """
    Simple detection rule:
    - if probability >= threshold for at least min_consecutive windows
      then declare MI detected
    Return:
    - detected (bool)
    - first_detect_idx (int or None)
    """
    count = 0

    for i, p in enumerate(smoothed_probs):
        if p >= threshold:
            count += 1
            if count >= min_consecutive:
                first_idx = i - min_consecutive + 1
                return True, first_idx
        else:
            count = 0

    return False, None



def run_one_subject(data_dir, subject_id):
    print("=" * 80)
    print(f"Pseudo-online MI vs Rest | subject {subject_id}")
    print(f"RUNS         = {RUNS}")
    print(f"TRAIN WINDOW = {TMIN}-{TMAX} s")
    print(f"BAND         = {L_FREQ}-{H_FREQ} Hz")
    print(f"TEST_SIZE    = {TEST_SIZE}")
    print(f"WIN_SEC      = {WIN_SEC}")
    print(f"STEP_SEC     = {STEP_SEC}")
    print(f"THRESHOLD    = {THRESHOLD}")
    print(f"SMOOTH_K     = {SMOOTH_K}")
    print(f"MIN_CONSEC   = {MIN_CONSECUTIVE_WINDOWS}")
    print("=" * 80)

    # 1) Load raw
    paths = load_subject_paths(str(data_dir), subject_id, RUNS)
    raw_full = load_subject_raw(paths)

    print(f"Filtering subject {subject_id}: {L_FREQ}-{H_FREQ} Hz")
    raw_filt = raw_full.copy().filter(
        L_FREQ,
        H_FREQ,
        fir_design="firwin",
        verbose="ERROR",
    )

    # 2) Build epochs using best window
    X, y, sfreq = make_epochs_from_filtered_raw(raw_filt, TMIN, TMAX)
    print(f"X shape = {X.shape}")
    print(f"Label counts = {np.bincount(y)}")
    print(f"Sampling frequency = {sfreq}")

    # 3) Train/test split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y))

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print("\nTrain/test split")
    print(f"Train X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"Test  X shape: {X_test.shape}, y shape: {y_test.shape}")

    # 4) Train model
    clf = build_classifier()
    clf.fit(X_train, y_train)

    # 5) Standard epoch-level evaluation first
    y_pred_epoch = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_epoch)
    bacc = balanced_accuracy_score(y_test, y_pred_epoch)
    f1 = f1_score(y_test, y_pred_epoch)
    cm = confusion_matrix(y_test, y_pred_epoch)

    print("\nEpoch-level test performance")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced accuracy: {bacc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    # 6) Pseudo-online replay
    trial_traces = []
    event_level_predictions = []
    event_level_truth = []

    for i in range(len(X_test)):
        X_epoch = X_test[i]
        true_label = int(y_test[i])

        windows, times_epoch_sec = sliding_windows_from_epoch(
            X_epoch,
            sfreq=sfreq,
            win_sec=WIN_SEC,
            step_sec=STEP_SEC,
        )

        if len(windows) == 0:
            continue

        probs = clf.predict_proba(windows)[:, 1]
        smoothed_probs = moving_average(probs, k=SMOOTH_K)

        detected, first_detect_idx = detect_from_probs(
            smoothed_probs,
            threshold=THRESHOLD,
            min_consecutive=MIN_CONSECUTIVE_WINDOWS,
        )

        pred_event = 1 if detected else 0

        # detection latency relative to epoch start and cue onset
        if detected and first_detect_idx is not None:
            detection_time_epoch_sec = float(times_epoch_sec[first_detect_idx])
            detection_time_cue_sec = float(times_epoch_sec[first_detect_idx] + TMIN)
        else:
            detection_time_epoch_sec = None
            detection_time_cue_sec = None

        trace_record = {
            "trial_index": int(i),
            "true_label": true_label,
            "times_epoch_sec": times_epoch_sec.tolist(),
            "times_cue_sec": (times_epoch_sec + TMIN).tolist(),
            "mi_probs": probs.tolist(),
            "smoothed_mi_probs": smoothed_probs.tolist(),
            "detected": bool(detected),
            "detection_time_epoch_sec": detection_time_epoch_sec,
            "detection_time_cue_sec": detection_time_cue_sec,
        }
        trial_traces.append(trace_record)

        event_level_predictions.append(pred_event)
        event_level_truth.append(true_label)

    # 7) Event-level pseudo-online metrics
    event_level_predictions = np.array(event_level_predictions, dtype=int)
    event_level_truth = np.array(event_level_truth, dtype=int)

    event_acc = accuracy_score(event_level_truth, event_level_predictions)
    event_bacc = balanced_accuracy_score(event_level_truth, event_level_predictions)
    event_f1 = f1_score(event_level_truth, event_level_predictions)
    event_cm = confusion_matrix(event_level_truth, event_level_predictions)

    # hit rate / false alarm rate
    mi_mask = event_level_truth == 1
    rest_mask = event_level_truth == 0

    hit_rate = float(np.mean(event_level_predictions[mi_mask] == 1)) if np.any(mi_mask) else None
    false_alarm_rate = float(np.mean(event_level_predictions[rest_mask] == 1)) if np.any(rest_mask) else None

    detection_latencies_epoch = [
        rec["detection_time_epoch_sec"]
        for rec in trial_traces
        if rec["true_label"] == 1 and rec["detected"] and rec["detection_time_epoch_sec"] is not None
    ]
    detection_latencies_cue = [
        rec["detection_time_cue_sec"]
        for rec in trial_traces
        if rec["true_label"] == 1 and rec["detected"] and rec["detection_time_cue_sec"] is not None
    ]

    mean_detection_latency_epoch = (
        float(np.mean(detection_latencies_epoch)) if len(detection_latencies_epoch) > 0 else None
    )
    mean_detection_latency_cue = (
        float(np.mean(detection_latencies_cue)) if len(detection_latencies_cue) > 0 else None
    )

    print("\nPseudo-online event-level performance")
    print(f"Event accuracy: {event_acc:.4f}")
    print(f"Event balanced accuracy: {event_bacc:.4f}")
    print(f"Event F1: {event_f1:.4f}")
    print(f"Event confusion matrix:\n{event_cm}")
    print(f"MI hit rate: {hit_rate}")
    print(f"Rest false alarm rate: {false_alarm_rate}")
    print(f"Mean detection latency relative to epoch start (MI trials): {mean_detection_latency_epoch}")
    print(f"Mean detection latency relative to cue onset (MI trials): {mean_detection_latency_cue}")

    subject_summary = {
        "subject_id": subject_id,
        "n_trials": int(len(X)),
        "label_counts": np.bincount(y).tolist(),
        "n_train_trials": int(len(X_train)),
        "n_test_trials": int(len(X_test)),
        "sampling_frequency": float(sfreq),
        "epoch_level_metrics": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
        },
        "pseudo_online_event_level_metrics": {
            "accuracy": float(event_acc),
            "balanced_accuracy": float(event_bacc),
            "f1": float(event_f1),
            "confusion_matrix": event_cm.tolist(),
            "mi_hit_rate": hit_rate,
            "rest_false_alarm_rate": false_alarm_rate,
            "mean_detection_latency_epoch_sec": mean_detection_latency_epoch,
            "mean_detection_latency_cue_sec": mean_detection_latency_cue,
        },
        "trial_traces": trial_traces,
    }

    return subject_summary



def main():
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_results = []
    epoch_accs = []
    event_accs = []
    hit_rates = []
    false_alarm_rates = []
    latencies_cue = []

    for subject_id in SUBJECT_IDS:
        subject_summary = run_one_subject(data_dir, subject_id)
        subject_results.append(subject_summary)

        epoch_accs.append(subject_summary["epoch_level_metrics"]["accuracy"])
        event_accs.append(subject_summary["pseudo_online_event_level_metrics"]["accuracy"])

        hr = subject_summary["pseudo_online_event_level_metrics"]["mi_hit_rate"]
        fa = subject_summary["pseudo_online_event_level_metrics"]["rest_false_alarm_rate"]
        lat = subject_summary["pseudo_online_event_level_metrics"]["mean_detection_latency_cue_sec"]

        if hr is not None:
            hit_rates.append(hr)
        if fa is not None:
            false_alarm_rates.append(fa)
        if lat is not None:
            latencies_cue.append(lat)

        per_subject_file = output_dir / f"pseudo_online_subject{subject_id}_bestwindow_0p5_3p5.json"
        with open(per_subject_file, "w", encoding="utf-8") as f:
            json.dump(subject_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved per-subject result to: {per_subject_file}\n")

    overall_summary = {
        "model": "PseudoOnline_Riemannian_TangentSpace_LR_MIvsRest_BestWindow",
        "subject_ids": SUBJECT_IDS,
        "num_subjects": len(SUBJECT_IDS),
        "runs": RUNS,
        "l_freq": L_FREQ,
        "h_freq": H_FREQ,
        "train_epoch_tmin": TMIN,
        "train_epoch_tmax": TMAX,
        "test_size": TEST_SIZE,
        "win_sec": WIN_SEC,
        "step_sec": STEP_SEC,
        "smooth_k": SMOOTH_K,
        "threshold": THRESHOLD,
        "min_consecutive_windows": MIN_CONSECUTIVE_WINDOWS,
        "subject_results": subject_results,
        "overall_mean_epoch_accuracy": float(np.mean(epoch_accs)) if len(epoch_accs) > 0 else None,
        "overall_mean_event_accuracy": float(np.mean(event_accs)) if len(event_accs) > 0 else None,
        "overall_mean_hit_rate": float(np.mean(hit_rates)) if len(hit_rates) > 0 else None,
        "overall_mean_false_alarm_rate": float(np.mean(false_alarm_rates)) if len(false_alarm_rates) > 0 else None,
        "overall_mean_detection_latency_cue_sec": float(np.mean(latencies_cue)) if len(latencies_cue) > 0 else None,
    }

    overall_file = output_dir / "pseudo_online_results_3subjects_bestwindow_0p5_3p5.json"
    with open(overall_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Finished all subjects")
    print(f"Overall mean epoch accuracy: {overall_summary['overall_mean_epoch_accuracy']}")
    print(f"Overall mean event accuracy: {overall_summary['overall_mean_event_accuracy']}")
    print(f"Overall mean hit rate: {overall_summary['overall_mean_hit_rate']}")
    print(f"Overall mean false alarm rate: {overall_summary['overall_mean_false_alarm_rate']}")
    print(f"Overall mean detection latency relative to cue onset: {overall_summary['overall_mean_detection_latency_cue_sec']}")
    print(f"Saved combined summary to: {overall_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
