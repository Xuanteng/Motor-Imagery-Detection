#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-online MI vs Rest detection with repeated train/val/test splits,
expanded window/step search, without FAR hard constraints.

What this script does:
1) For each subject, repeat train/val/test splitting multiple times
2) In each repeat, use validation only to choose pseudo-online parameters
3) No FAR hard constraint is used during model selection
4) Retrain on train+val with the chosen configuration
5) Evaluate once on the held-out test set
6) Save per-subject and combined JSON summaries with mean/std metrics

Compared with the FAR-constrained version:
- Keeps stricter causal latency by default: WINDOW_TIME_REFERENCE = "end"
- Keeps the expanded win_sec / step_sec search space
- Removes hard FAR feasibility filtering, so you can inspect the natural accuracy / FAR / latency tradeoff
- This copy is configured for subjects 1-10; missing files are downloaded by MNE

April 17, 2026, unconstrained expanded-search repeated-split version
"""

from pathlib import Path
import json
import itertools
from collections import Counter
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

SUBJECT_IDS = list(range(1, 11))  # subjects 1-10
RUNS = [4, 8, 12]

L_FREQ = 8.0
H_FREQ = 30.0

# Candidate training epoch windows.
TRAIN_WINDOWS = [
    (0.5, 3.5),
    # (1.0, 4.0),
    # (1.0, 4.5),
]

BASE_RANDOM_STATE = 42
N_REPEATS = 10

TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# Expanded pseudo-online search space.
WIN_SEC_LIST = [0.5, 0.75, 1.0]
STEP_SEC_LIST = [0.1, 0.125, 0.25]

# Candidate decision-layer parameters.
SMOOTH_K_LIST = [1, 3, 5]
THRESHOLD_LIST = [0.5, 0.6, 0.7]
MIN_CONSECUTIVE_LIST = [1, 2, 3]

# Use stricter causal latency by default.
WINDOW_TIME_REFERENCE = "end"

# Unconstrained selection: no hard FAR / hit / latency thresholds.

# JSON controls.
NUM_TOP_VAL_RESULTS_TO_SAVE = 10
SAVE_TEST_TRIAL_TRACES = False
MAX_REPEATS_WITH_TRACES = 1


# =========================
# Helper functions
# =========================
def load_subject_paths(data_dir, subject_id, runs):
    """Load EEGBCI file paths for one subject.

    MNE will reuse local files already present under DATA_DIR and download any
    missing subject/run files automatically if internet access is available.
    """
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
    y = np.where(event_codes == 1, 0, 1).astype(np.int64)

    return X, y, epochs.info["sfreq"]



def build_classifier(random_state):
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
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



def sliding_windows_from_epoch(
    X_epoch,
    sfreq,
    win_sec=1.0,
    step_sec=0.25,
    time_reference="end",
):
    """
    X_epoch: (n_channels, n_times)

    Return:
        windows: (n_windows, n_channels, win_samples)
        times_epoch_sec: per-window times relative to epoch start
            - center: window center times
            - end: window end times
    """
    n_channels, n_times = X_epoch.shape
    win_samples = int(round(win_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))

    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("win_sec and step_sec must both be > 0")
    if time_reference not in {"center", "end"}:
        raise ValueError("time_reference must be either 'center' or 'end'")

    windows = []
    times_epoch_sec = []

    start = 0
    while start + win_samples <= n_times:
        end = start + win_samples
        windows.append(X_epoch[:, start:end])

        if time_reference == "center":
            ref_sample = start + win_samples / 2.0
        else:
            ref_sample = end

        ref_time = ref_sample / sfreq
        times_epoch_sec.append(ref_time)
        start += step_samples

    if len(windows) > 0:
        windows = np.stack(windows, axis=0)
    else:
        windows = np.empty((0, n_channels, win_samples), dtype=float)

    return windows, np.asarray(times_epoch_sec, dtype=float)



def detect_from_probs(smoothed_probs, threshold=0.6, min_consecutive=2):
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



def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }



def split_train_val_test_indices(
    y,
    train_size,
    val_size,
    test_size,
    random_state,
):
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    all_idx = np.arange(len(y))

    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_val_idx, test_idx = next(splitter_test.split(all_idx, y))

    y_train_val = y[train_val_idx]
    rel_val_size = val_size / (train_size + val_size)

    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=rel_val_size,
        random_state=random_state + 1,
    )
    train_rel_idx, val_rel_idx = next(splitter_val.split(train_val_idx, y_train_val))

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx



def replay_trials_collect_base_traces(
    clf,
    X_epochs,
    y_epochs,
    sfreq,
    tmin,
    win_sec,
    step_sec,
    time_reference,
):
    base_traces = []

    for i in range(len(X_epochs)):
        X_epoch = X_epochs[i]
        true_label = int(y_epochs[i])

        windows, times_epoch_sec = sliding_windows_from_epoch(
            X_epoch,
            sfreq=sfreq,
            win_sec=win_sec,
            step_sec=step_sec,
            time_reference=time_reference,
        )

        if len(windows) == 0:
            continue

        probs = clf.predict_proba(windows)[:, 1]

        base_traces.append(
            {
                "trial_index": int(i),
                "true_label": true_label,
                "times_epoch_sec": times_epoch_sec.tolist(),
                "times_cue_sec": (times_epoch_sec + tmin).tolist(),
                "mi_probs": probs.tolist(),
            }
        )

    return base_traces



def evaluate_decision_layer_from_base_traces(
    base_traces,
    smooth_k,
    threshold,
    min_consecutive,
    include_trial_traces=False,
):
    event_level_predictions = []
    event_level_truth = []
    trial_results = []

    for rec in base_traces:
        probs = np.asarray(rec["mi_probs"], dtype=float)
        smoothed_probs = moving_average(probs, k=smooth_k)

        detected, first_detect_idx = detect_from_probs(
            smoothed_probs,
            threshold=threshold,
            min_consecutive=min_consecutive,
        )

        pred_event = 1 if detected else 0
        true_label = int(rec["true_label"])

        if detected and first_detect_idx is not None:
            detection_time_epoch_sec = float(rec["times_epoch_sec"][first_detect_idx])
            detection_time_cue_sec = float(rec["times_cue_sec"][first_detect_idx])
        else:
            detection_time_epoch_sec = None
            detection_time_cue_sec = None

        event_level_predictions.append(pred_event)
        event_level_truth.append(true_label)

        if include_trial_traces:
            trial_results.append(
                {
                    "trial_index": int(rec["trial_index"]),
                    "true_label": true_label,
                    "times_epoch_sec": rec["times_epoch_sec"],
                    "times_cue_sec": rec["times_cue_sec"],
                    "mi_probs": rec["mi_probs"],
                    "smoothed_mi_probs": smoothed_probs.tolist(),
                    "detected": bool(detected),
                    "detection_time_epoch_sec": detection_time_epoch_sec,
                    "detection_time_cue_sec": detection_time_cue_sec,
                }
            )

    metric_core = compute_basic_metrics(event_level_truth, event_level_predictions)

    event_level_truth = np.asarray(event_level_truth, dtype=int)
    event_level_predictions = np.asarray(event_level_predictions, dtype=int)

    mi_mask = event_level_truth == 1
    rest_mask = event_level_truth == 0

    hit_rate = float(np.mean(event_level_predictions[mi_mask] == 1)) if np.any(mi_mask) else None
    false_alarm_rate = (
        float(np.mean(event_level_predictions[rest_mask] == 1)) if np.any(rest_mask) else None
    )

    if include_trial_traces:
        source_for_latency = trial_results
    else:
        source_for_latency = []
        for truth, pred, rec in zip(event_level_truth, event_level_predictions, base_traces):
            if truth == 1 and pred == 1:
                smoothed_probs = moving_average(np.asarray(rec["mi_probs"], dtype=float), k=smooth_k)
                detected, first_detect_idx = detect_from_probs(
                    smoothed_probs,
                    threshold=threshold,
                    min_consecutive=min_consecutive,
                )
                if detected and first_detect_idx is not None:
                    source_for_latency.append(
                        {
                            "true_label": int(truth),
                            "detected": True,
                            "detection_time_epoch_sec": float(rec["times_epoch_sec"][first_detect_idx]),
                            "detection_time_cue_sec": float(rec["times_cue_sec"][first_detect_idx]),
                        }
                    )

    detection_latencies_epoch = [
        rec["detection_time_epoch_sec"]
        for rec in source_for_latency
        if rec["true_label"] == 1 and rec["detected"] and rec["detection_time_epoch_sec"] is not None
    ]
    detection_latencies_cue = [
        rec["detection_time_cue_sec"]
        for rec in source_for_latency
        if rec["true_label"] == 1 and rec["detected"] and rec["detection_time_cue_sec"] is not None
    ]

    mean_detection_latency_epoch = (
        float(np.mean(detection_latencies_epoch)) if len(detection_latencies_epoch) > 0 else None
    )
    mean_detection_latency_cue = (
        float(np.mean(detection_latencies_cue)) if len(detection_latencies_cue) > 0 else None
    )

    metrics = {
        **metric_core,
        "mi_hit_rate": hit_rate,
        "rest_false_alarm_rate": false_alarm_rate,
        "mean_detection_latency_epoch_sec": mean_detection_latency_epoch,
        "mean_detection_latency_cue_sec": mean_detection_latency_cue,
    }

    return metrics, trial_results



def get_epochs_for_window(raw_filt, tmin, tmax, y_reference=None):
    X, y, sfreq = make_epochs_from_filtered_raw(raw_filt, tmin=tmin, tmax=tmax)
    if y_reference is not None and not np.array_equal(y, y_reference):
        raise RuntimeError(
            f"Label order mismatch for window ({tmin}, {tmax}). "
            "Please make sure all candidate windows produce the same trial order."
        )
    return X, y, sfreq



def compact_result_for_json(result_dict):
    out = dict(result_dict)
    out.pop("trial_results", None)
    out.pop("base_traces", None)
    return out



def metric_mean_std_from_repeat_summaries(repeat_summaries, key_path):
    values = []
    for rep in repeat_summaries:
        d = rep
        for key in key_path:
            d = d[key]
        if d is not None:
            values.append(float(d))

    if len(values) == 0:
        return {"mean": None, "std": None, "n": 0}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "n": int(len(values)),
    }



def config_to_string(best_config):
    return (
        f"window=({best_config['train_epoch_tmin']},{best_config['train_epoch_tmax']})|"
        f"win={best_config['win_sec']}|step={best_config['step_sec']}|"
        f"smooth_k={best_config['smooth_k']}|thr={best_config['threshold']}|"
        f"min_consec={best_config['min_consecutive_windows']}"
    )



def unconstrained_sort_key(metrics):
    """
    Higher is better.

    Priority:
    1) event balanced accuracy
    2) lower false alarm rate
    3) higher hit rate
    4) lower latency
    5) event F1
    """
    far = metrics["rest_false_alarm_rate"]
    hit = metrics["mi_hit_rate"]
    lat = metrics["mean_detection_latency_cue_sec"]

    if far is None:
        far = 1.0
    if hit is None:
        hit = 0.0
    if lat is None:
        lat = 1e9

    return (
        metrics["balanced_accuracy"],
        -far,
        hit,
        -lat,
        metrics["f1"],
    )



def select_best_validation_result(val_search_results):
    val_search_results.sort(
        key=lambda d: unconstrained_sort_key(d["pseudo_online_val_event_metrics"]),
        reverse=True,
    )
    best_result = val_search_results[0]
    selection_status = "unconstrained_best"
    return best_result, selection_status



def run_one_subject(data_dir, subject_id):

    print("=" * 80)
    print(f"Pseudo-online repeated unconstrained expanded-search train/val/test | subject {subject_id}")
    print(f"RUNS                    = {RUNS}")
    print(f"BAND                    = {L_FREQ}-{H_FREQ} Hz")
    print(f"N_REPEATS               = {N_REPEATS}")
    print(f"TRAIN_SIZE / VAL / TEST = {TRAIN_SIZE} / {VAL_SIZE} / {TEST_SIZE}")
    print(f"TRAIN_WINDOWS           = {TRAIN_WINDOWS}")
    print(f"WIN_SEC_LIST            = {WIN_SEC_LIST}")
    print(f"STEP_SEC_LIST           = {STEP_SEC_LIST}")
    print(f"SMOOTH_K_LIST           = {SMOOTH_K_LIST}")
    print(f"THRESHOLD_LIST          = {THRESHOLD_LIST}")
    print(f"MIN_CONSEC_LIST         = {MIN_CONSECUTIVE_LIST}")
    print(f"TIME_REFERENCE          = {WINDOW_TIME_REFERENCE}")
    print("=" * 80)

    paths = load_subject_paths(str(data_dir), subject_id, RUNS)
    raw_full = load_subject_raw(paths)

    print(f"Filtering subject {subject_id}: {L_FREQ}-{H_FREQ} Hz")
    raw_filt = raw_full.copy().filter(
        L_FREQ,
        H_FREQ,
        fir_design="firwin",
        verbose="ERROR",
    )

    ref_tmin = min(t[0] for t in TRAIN_WINDOWS)
    ref_tmax = max(t[1] for t in TRAIN_WINDOWS)
    X_ref, y_ref, sfreq_ref = make_epochs_from_filtered_raw(raw_filt, ref_tmin, ref_tmax)

    print(f"Reference X shape = {X_ref.shape}")
    print(f"Reference label counts = {np.bincount(y_ref)}")
    print(f"Sampling frequency = {sfreq_ref}")

    epoch_cache = {}
    for tmin, tmax in TRAIN_WINDOWS:
        X_all, y_all, sfreq = get_epochs_for_window(raw_filt, tmin, tmax, y_reference=y_ref)
        epoch_cache[(tmin, tmax)] = (X_all, y_all, sfreq)

    repeat_summaries = []

    for repeat_idx in range(N_REPEATS):
        repeat_seed = BASE_RANDOM_STATE + 100 * repeat_idx
        print("-" * 80)
        print(f"Subject {subject_id} | repeat {repeat_idx + 1}/{N_REPEATS} | seed={repeat_seed}")

        train_idx, val_idx, test_idx = split_train_val_test_indices(
            y_ref,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            random_state=repeat_seed,
        )

        val_search_results = []

        for tmin, tmax in TRAIN_WINDOWS:
            X_all, y_all, sfreq = epoch_cache[(tmin, tmax)]

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]

            clf = build_classifier(random_state=repeat_seed)
            clf.fit(X_train, y_train)

            y_val_pred_epoch = clf.predict(X_val)
            epoch_val_metrics = compute_basic_metrics(y_val, y_val_pred_epoch)

            for win_sec, step_sec in itertools.product(WIN_SEC_LIST, STEP_SEC_LIST):
                base_traces = replay_trials_collect_base_traces(
                    clf=clf,
                    X_epochs=X_val,
                    y_epochs=y_val,
                    sfreq=sfreq,
                    tmin=tmin,
                    win_sec=win_sec,
                    step_sec=step_sec,
                    time_reference=WINDOW_TIME_REFERENCE,
                )

                for smooth_k, threshold, min_consecutive in itertools.product(
                    SMOOTH_K_LIST,
                    THRESHOLD_LIST,
                    MIN_CONSECUTIVE_LIST,
                ):
                    event_metrics, _ = evaluate_decision_layer_from_base_traces(
                        base_traces=base_traces,
                        smooth_k=smooth_k,
                        threshold=threshold,
                        min_consecutive=min_consecutive,
                        include_trial_traces=False,
                    )

                    
                    result = {
                        "subject_id": int(subject_id),
                        "repeat_index": int(repeat_idx),
                        "split_seed": int(repeat_seed),
                        "train_epoch_tmin": float(tmin),
                        "train_epoch_tmax": float(tmax),
                        "win_sec": float(win_sec),
                        "step_sec": float(step_sec),
                        "smooth_k": int(smooth_k),
                        "threshold": float(threshold),
                        "min_consecutive_windows": int(min_consecutive),
                        "epoch_level_val_metrics": epoch_val_metrics,
                        "pseudo_online_val_event_metrics": event_metrics,
                    }
                    val_search_results.append(result)

        best_val_result, selection_status = select_best_validation_result(val_search_results)


        best_tmin = best_val_result["train_epoch_tmin"]
        best_tmax = best_val_result["train_epoch_tmax"]
        best_win_sec = best_val_result["win_sec"]
        best_step_sec = best_val_result["step_sec"]
        best_smooth_k = best_val_result["smooth_k"]
        best_threshold = best_val_result["threshold"]
        best_min_consecutive = best_val_result["min_consecutive_windows"]

        X_best, y_best, sfreq_best = epoch_cache[(best_tmin, best_tmax)]

        train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))
        X_train_val = X_best[train_val_idx]
        y_train_val = y_best[train_val_idx]
        X_test = X_best[test_idx]
        y_test = y_best[test_idx]

        clf_final = build_classifier(random_state=repeat_seed)
        clf_final.fit(X_train_val, y_train_val)

        y_test_pred_epoch = clf_final.predict(X_test)
        epoch_test_metrics = compute_basic_metrics(y_test, y_test_pred_epoch)

        test_base_traces = replay_trials_collect_base_traces(
            clf=clf_final,
            X_epochs=X_test,
            y_epochs=y_test,
            sfreq=sfreq_best,
            tmin=best_tmin,
            win_sec=best_win_sec,
            step_sec=best_step_sec,
            time_reference=WINDOW_TIME_REFERENCE,
        )

        should_save_traces = SAVE_TEST_TRIAL_TRACES and repeat_idx < MAX_REPEATS_WITH_TRACES
        pseudo_online_test_metrics, test_trial_results = evaluate_decision_layer_from_base_traces(
            base_traces=test_base_traces,
            smooth_k=best_smooth_k,
            threshold=best_threshold,
            min_consecutive=best_min_consecutive,
            include_trial_traces=should_save_traces,
        )

        repeat_summary = {
            "repeat_index": int(repeat_idx),
            "split_seed": int(repeat_seed),
            "split_counts": {
                "n_train_trials": int(len(train_idx)),
                "n_val_trials": int(len(val_idx)),
                "n_test_trials": int(len(test_idx)),
                "n_train_plus_val_trials": int(len(train_val_idx)),
            },
            "selection_status": selection_status,
            "best_validation_config": {
                "train_epoch_tmin": float(best_tmin),
                "train_epoch_tmax": float(best_tmax),
                "win_sec": float(best_win_sec),
                "step_sec": float(best_step_sec),
                "smooth_k": int(best_smooth_k),
                "threshold": float(best_threshold),
                "min_consecutive_windows": int(best_min_consecutive),
                "epoch_level_val_metrics": best_val_result["epoch_level_val_metrics"],
                "pseudo_online_val_event_metrics": best_val_result["pseudo_online_val_event_metrics"],
            },
            "held_out_test_epoch_level_metrics": epoch_test_metrics,
            "held_out_test_pseudo_online_event_level_metrics": pseudo_online_test_metrics,
            "top_validation_results": [
                compact_result_for_json(r) for r in val_search_results[:NUM_TOP_VAL_RESULTS_TO_SAVE]
            ],
        }

        if should_save_traces:
            repeat_summary["held_out_test_trial_traces"] = test_trial_results

        repeat_summaries.append(repeat_summary)

        print(
            f"Repeat {repeat_idx + 1}: status={selection_status}, "
            f"event_acc={pseudo_online_test_metrics['accuracy']:.4f}, "
            f"hit={pseudo_online_test_metrics['mi_hit_rate']}, "
            f"far={pseudo_online_test_metrics['rest_false_alarm_rate']}, "
            f"lat={pseudo_online_test_metrics['mean_detection_latency_cue_sec']}"
        )

    best_config_counter = Counter(
        config_to_string(rep["best_validation_config"]) for rep in repeat_summaries
    )
    selection_status_counter = Counter(rep["selection_status"] for rep in repeat_summaries)

    aggregate_test_metrics = {
        "epoch_accuracy": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_epoch_level_metrics", "accuracy"]
        ),
        "epoch_balanced_accuracy": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_epoch_level_metrics", "balanced_accuracy"]
        ),
        "epoch_f1": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_epoch_level_metrics", "f1"]
        ),
        "event_accuracy": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_pseudo_online_event_level_metrics", "accuracy"]
        ),
        "event_balanced_accuracy": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_pseudo_online_event_level_metrics", "balanced_accuracy"]
        ),
        "event_f1": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_pseudo_online_event_level_metrics", "f1"]
        ),
        "hit_rate": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_pseudo_online_event_level_metrics", "mi_hit_rate"]
        ),
        "false_alarm_rate": metric_mean_std_from_repeat_summaries(
            repeat_summaries, ["held_out_test_pseudo_online_event_level_metrics", "rest_false_alarm_rate"]
        ),
        "detection_latency_epoch_sec": metric_mean_std_from_repeat_summaries(
            repeat_summaries,
            ["held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_epoch_sec"],
        ),
        "detection_latency_cue_sec": metric_mean_std_from_repeat_summaries(
            repeat_summaries,
            ["held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_cue_sec"],
        ),
    }

    subject_summary = {
        "subject_id": int(subject_id),
        "runs": RUNS,
        "l_freq": float(L_FREQ),
        "h_freq": float(H_FREQ),
        "time_reference": WINDOW_TIME_REFERENCE,
        "num_trials_total": int(len(y_ref)),
        "label_counts_total": np.bincount(y_ref).tolist(),
        "sampling_frequency": float(sfreq_ref),
        "n_repeats": int(N_REPEATS),
        "split_sizes": {
            "train_size": float(TRAIN_SIZE),
            "val_size": float(VAL_SIZE),
            "test_size": float(TEST_SIZE),
        },
                "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
        },
        "aggregate_test_metrics_across_repeats": aggregate_test_metrics,
        "selection_status_frequency": [
            {"status": status, "count": int(cnt)}
            for status, cnt in selection_status_counter.most_common()
        ],
        "best_config_frequency": [
            {"config": cfg, "count": int(cnt)} for cfg, cnt in best_config_counter.most_common()
        ],
        "repeat_results": repeat_summaries,
    }

    return subject_summary



def summarize_overall(subject_results):
    subject_event_acc_means = []
    subject_hit_means = []
    subject_far_means = []
    subject_lat_cue_means = []

    flat_event_acc = []
    flat_hit = []
    flat_far = []
    flat_lat_cue = []

    total_unconstrained = 0

    for subj in subject_results:
        m = subj["aggregate_test_metrics_across_repeats"]
        if m["event_accuracy"]["mean"] is not None:
            subject_event_acc_means.append(m["event_accuracy"]["mean"])
        if m["hit_rate"]["mean"] is not None:
            subject_hit_means.append(m["hit_rate"]["mean"])
        if m["false_alarm_rate"]["mean"] is not None:
            subject_far_means.append(m["false_alarm_rate"]["mean"])
        if m["detection_latency_cue_sec"]["mean"] is not None:
            subject_lat_cue_means.append(m["detection_latency_cue_sec"]["mean"])

        for rep in subj["repeat_results"]:
            rep_m = rep["held_out_test_pseudo_online_event_level_metrics"]
            if rep_m["accuracy"] is not None:
                flat_event_acc.append(rep_m["accuracy"])
            if rep_m["mi_hit_rate"] is not None:
                flat_hit.append(rep_m["mi_hit_rate"])
            if rep_m["rest_false_alarm_rate"] is not None:
                flat_far.append(rep_m["rest_false_alarm_rate"])
            if rep_m["mean_detection_latency_cue_sec"] is not None:
                flat_lat_cue.append(rep_m["mean_detection_latency_cue_sec"])

            total_unconstrained += 1

    def mean_std(xs):
        if len(xs) == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=0)), "n": int(len(xs))}

    return {
        "selection_status_totals": {
            "unconstrained_best": int(total_unconstrained),
        },
        "overall_mean_of_subject_means": {
            "event_accuracy": mean_std(subject_event_acc_means),
            "hit_rate": mean_std(subject_hit_means),
            "false_alarm_rate": mean_std(subject_far_means),
            "detection_latency_cue_sec": mean_std(subject_lat_cue_means),
        },
        "overall_across_all_subject_repeat_pairs": {
            "event_accuracy": mean_std(flat_event_acc),
            "hit_rate": mean_std(flat_hit),
            "false_alarm_rate": mean_std(flat_far),
            "detection_latency_cue_sec": mean_std(flat_lat_cue),
        },
    }



def selected_config_row(subject_id, repeat_summary):
    cfg = repeat_summary["best_validation_config"]
    testm = repeat_summary["held_out_test_pseudo_online_event_level_metrics"]
    valm = cfg["pseudo_online_val_event_metrics"]
    return {
        "subject_id": int(subject_id),
        "repeat_index": int(repeat_summary["repeat_index"]),
        "split_seed": int(repeat_summary["split_seed"]),
        "selection_status": repeat_summary["selection_status"],
        "train_epoch_tmin": float(cfg["train_epoch_tmin"]),
        "train_epoch_tmax": float(cfg["train_epoch_tmax"]),
        "win_sec": float(cfg["win_sec"]),
        "step_sec": float(cfg["step_sec"]),
        "smooth_k": int(cfg["smooth_k"]),
        "threshold": float(cfg["threshold"]),
        "min_consecutive_windows": int(cfg["min_consecutive_windows"]),
        "val_event_accuracy": valm["accuracy"],
        "val_hit_rate": valm["mi_hit_rate"],
        "val_false_alarm_rate": valm["rest_false_alarm_rate"],
        "val_latency_cue_sec": valm["mean_detection_latency_cue_sec"],
        "test_event_accuracy": testm["accuracy"],
        "test_hit_rate": testm["mi_hit_rate"],
        "test_false_alarm_rate": testm["rest_false_alarm_rate"],
        "test_latency_cue_sec": testm["mean_detection_latency_cue_sec"],
    }


def export_selected_config_csv(subject_results, output_csv):
    import csv
    rows = []
    for subj in subject_results:
        sid = subj["subject_id"]
        for rep in subj["repeat_results"]:
            rows.append(selected_config_row(sid, rep))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_results = []

    for subject_id in SUBJECT_IDS:
        subject_summary = run_one_subject(data_dir, subject_id)
        subject_results.append(subject_summary)

        per_subject_file = output_dir / f"pseudo_online_repeated_expanded_search_unconstrained_subject{subject_id}.json"
        with open(per_subject_file, "w", encoding="utf-8") as f:
            json.dump(subject_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved per-subject result to: {per_subject_file}\n")

    overall_summary = {
        "model": "PseudoOnline_Riemannian_TangentSpace_LR_MIvsRest_RepeatedTrainValTest_ExpandedSearch_Unconstrained",
        "subject_ids": SUBJECT_IDS,
        "num_subjects": len(SUBJECT_IDS),
        "runs": RUNS,
        "l_freq": L_FREQ,
        "h_freq": H_FREQ,
        "n_repeats": N_REPEATS,
        "base_random_state": BASE_RANDOM_STATE,
        "split_sizes": {
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
        },
                "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
        },
        "time_reference": WINDOW_TIME_REFERENCE,
        "subject_results": subject_results,
        "overall_summary_metrics": summarize_overall(subject_results),
    }

    overall_file = output_dir / "pseudo_online_repeated_expanded_search_unconstrained_results.json"
    with open(overall_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    selected_csv = output_dir / "pseudo_online_repeated_expanded_search_unconstrained_selected_configs.csv"
    export_selected_config_csv(subject_results, selected_csv)

    overall_pair_metrics = overall_summary["overall_summary_metrics"]["overall_across_all_subject_repeat_pairs"]
    overall_status = overall_summary["overall_summary_metrics"]["selection_status_totals"]

    print("=" * 80)
    print("Finished all subjects")
    print(f"Overall event accuracy mean: {overall_pair_metrics['event_accuracy']['mean']}")
    print(f"Overall hit rate mean: {overall_pair_metrics['hit_rate']['mean']}")
    print(f"Overall false alarm rate mean: {overall_pair_metrics['false_alarm_rate']['mean']}")
    print(f"Overall latency cue mean: {overall_pair_metrics['detection_latency_cue_sec']['mean']}")
    print(f"Selection status totals: {overall_status}")
    print(f"Saved combined summary to: {overall_file}")
    print(f"Saved selected-config CSV to: {selected_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
