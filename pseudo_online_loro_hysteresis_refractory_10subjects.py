#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-online MI vs Rest detection with leave-one-run-out (LORO),
10 subjects, expanded window/step search, and a state-machine decision layer.

New decision-layer features:
- dual threshold / hysteresis:
    * detector arms when smoothed probability first exceeds `threshold` (high threshold)
    * detector stays valid while subsequent windows remain above `low_threshold`
- refractory period:
    * after a trigger, additional triggers are suppressed for `refractory_sec`
- arm-after-cue time:
    * detector is not allowed to trigger before `arm_after_cue_sec`

Important note:
This script still reports one event decision per epoch / trial.
In that setting, dual-threshold and arm-after-cue usually matter more than
refractory period, because only the first trigger in each trial affects the
trial-level decision. Refractory is included anyway because it becomes useful
when you later move toward more continuous online replay.

April 20, 2026
Author: XYan
"""

from pathlib import Path
import json
import csv
import itertools
from collections import Counter, defaultdict
import numpy as np
import mne
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# =========================
# User settings
# =========================
DATA_DIR = "/Users/xuantengyan/Library/CloudStorage/OneDrive-Personal/Others/Jobs/Ton Bridge Medical Device/EEGBCI_Data"
OUTPUT_DIR = "/Users/xuantengyan/Library/CloudStorage/OneDrive-Personal/Others/Jobs/Ton Bridge Medical Device/EEGBCI_Results"

SUBJECT_IDS = list(range(1, 11))
RUNS = [4, 8, 12]

L_FREQ = 8.0
H_FREQ = 30.0

TRAIN_WINDOWS = [
    (0.5, 3.5),
]

WIN_SEC_LIST = [0.5, 0.75, 1.0]
STEP_SEC_LIST = [0.1, 0.125, 0.25]
SMOOTH_K_LIST = [1, 3, 5, 7]

# High threshold (same meaning as the old `threshold`).
THRESHOLD_LIST = [0.5, 0.6, 0.7, 0.75, 0.8]
# Low threshold used after the detector has armed.
LOW_THRESHOLD_LIST = [0.4, 0.5, 0.6, 0.7]
MIN_CONSECUTIVE_LIST = [1, 2, 3, 4]

# New timing controls.
REFRACTORY_SEC_LIST = [0.0, 0.25, 0.5]
ARM_AFTER_CUE_SEC_LIST = [0.0, 0.25, 0.5]

WINDOW_TIME_REFERENCE = "end"
NUM_TOP_INNER_VAL_RESULTS_TO_SAVE = 10
SAVE_TEST_TRIAL_TRACES = False


# =========================
# Helpers
# =========================
def load_run_path(data_dir, subject_id, run_id):
    paths = mne.datasets.eegbci.load_data(subject_id, [run_id], path=str(data_dir))
    return paths[0]



def load_run_raw(edf_path):
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        stim_channel="auto",
        verbose="ERROR",
    )
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
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

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
    return X, y, float(epochs.info["sfreq"])



def build_classifier(random_state=42):
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



def sliding_windows_from_epoch(X_epoch, sfreq, win_sec=1.0, step_sec=0.25, time_reference="end"):
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
        ref_sample = start + win_samples / 2.0 if time_reference == "center" else end
        times_epoch_sec.append(ref_sample / sfreq)
        start += step_samples

    if len(windows) > 0:
        windows = np.stack(windows, axis=0)
    else:
        windows = np.empty((0, n_channels, win_samples), dtype=float)

    return windows, np.asarray(times_epoch_sec, dtype=float)



def run_hysteresis_detector(
    smoothed_probs,
    times_epoch_sec,
    high_threshold=0.6,
    low_threshold=0.5,
    min_consecutive=2,
    refractory_sec=0.0,
    arm_after_cue_sec=0.0,
):
    """
    State-machine detector.

    Rules:
    - ignore windows before arm_after_cue_sec
    - arm only when probability first exceeds high_threshold
    - once armed, allow the run to continue while probability stays >= low_threshold
    - trigger after min_consecutive armed windows
    - after each trigger, suppress re-triggering for refractory_sec

    Returns
    -------
    detected : bool
    first_trigger_idx : int or None
    trigger_indices : list[int]
    """
    probs = np.asarray(smoothed_probs, dtype=float)
    times = np.asarray(times_epoch_sec, dtype=float)

    if len(probs) != len(times):
        raise ValueError("smoothed_probs and times_epoch_sec must have the same length")
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must be <= high_threshold")
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be >= 1")

    armed = False
    candidate_start_idx = None
    armed_count = 0
    refractory_until = -np.inf
    trigger_indices = []

    for i, (p, t) in enumerate(zip(probs, times)):
        if t < arm_after_cue_sec:
            armed = False
            candidate_start_idx = None
            armed_count = 0
            continue

        if t < refractory_until:
            armed = False
            candidate_start_idx = None
            armed_count = 0
            continue

        if not armed:
            if p >= high_threshold:
                armed = True
                candidate_start_idx = i
                armed_count = 1
                if armed_count >= min_consecutive:
                    trigger_indices.append(candidate_start_idx)
                    refractory_until = t + refractory_sec
                    armed = False
                    candidate_start_idx = None
                    armed_count = 0
            else:
                candidate_start_idx = None
                armed_count = 0
        else:
            if p >= low_threshold:
                armed_count += 1
                if armed_count >= min_consecutive:
                    trigger_indices.append(candidate_start_idx)
                    refractory_until = t + refractory_sec
                    armed = False
                    candidate_start_idx = None
                    armed_count = 0
            else:
                armed = False
                candidate_start_idx = None
                armed_count = 0

    detected = len(trigger_indices) > 0
    first_trigger_idx = trigger_indices[0] if detected else None
    return detected, first_trigger_idx, trigger_indices



def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }



def replay_trials_collect_base_traces(clf, X_epochs, y_epochs, sfreq, tmin, win_sec, step_sec, time_reference):
    base_traces = []
    for i in range(len(X_epochs)):
        windows, times_epoch_sec = sliding_windows_from_epoch(
            X_epochs[i], sfreq=sfreq, win_sec=win_sec, step_sec=step_sec, time_reference=time_reference
        )
        if len(windows) == 0:
            continue
        probs = clf.predict_proba(windows)[:, 1]
        base_traces.append(
            {
                "trial_index": int(i),
                "true_label": int(y_epochs[i]),
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
    low_threshold,
    min_consecutive,
    refractory_sec,
    arm_after_cue_sec,
    include_trial_traces=False,
):
    event_preds, event_truth = [], []
    trial_results = []

    for rec in base_traces:
        probs = np.asarray(rec["mi_probs"], dtype=float)
        times_epoch_sec = np.asarray(rec["times_epoch_sec"], dtype=float)
        times_cue_sec = np.asarray(rec["times_cue_sec"], dtype=float)
        smoothed = moving_average(probs, k=smooth_k)

        detected, first_detect_idx, trigger_indices = run_hysteresis_detector(
            smoothed_probs=smoothed,
            times_epoch_sec=times_epoch_sec,
            high_threshold=threshold,
            low_threshold=low_threshold,
            min_consecutive=min_consecutive,
            refractory_sec=refractory_sec,
            arm_after_cue_sec=arm_after_cue_sec,
        )

        pred_event = 1 if detected else 0
        true_label = int(rec["true_label"])

        if detected and first_detect_idx is not None:
            det_epoch = float(times_epoch_sec[first_detect_idx])
            det_cue = float(times_cue_sec[first_detect_idx])
        else:
            det_epoch = None
            det_cue = None

        event_preds.append(pred_event)
        event_truth.append(true_label)

        if include_trial_traces:
            trial_results.append(
                {
                    "trial_index": int(rec["trial_index"]),
                    "true_label": true_label,
                    "times_epoch_sec": rec["times_epoch_sec"],
                    "times_cue_sec": rec["times_cue_sec"],
                    "mi_probs": rec["mi_probs"],
                    "smoothed_mi_probs": smoothed.tolist(),
                    "detected": bool(detected),
                    "detection_time_epoch_sec": det_epoch,
                    "detection_time_cue_sec": det_cue,
                    "num_triggers": int(len(trigger_indices)),
                    "trigger_indices": [int(x) for x in trigger_indices],
                }
            )

    metric_core = compute_basic_metrics(event_truth, event_preds)
    event_truth = np.asarray(event_truth, dtype=int)
    event_preds = np.asarray(event_preds, dtype=int)

    mi_mask = event_truth == 1
    rest_mask = event_truth == 0
    hit_rate = float(np.mean(event_preds[mi_mask] == 1)) if np.any(mi_mask) else None
    far = float(np.mean(event_preds[rest_mask] == 1)) if np.any(rest_mask) else None

    detection_latencies_epoch = []
    detection_latencies_cue = []
    if include_trial_traces:
        source = trial_results
        for tr in source:
            if tr["true_label"] == 1 and tr["detected"] and tr["detection_time_epoch_sec"] is not None:
                detection_latencies_epoch.append(float(tr["detection_time_epoch_sec"]))
                detection_latencies_cue.append(float(tr["detection_time_cue_sec"]))
    else:
        for rec, truth in zip(base_traces, event_truth):
            if truth != 1:
                continue
            probs = np.asarray(rec["mi_probs"], dtype=float)
            times_epoch_sec = np.asarray(rec["times_epoch_sec"], dtype=float)
            times_cue_sec = np.asarray(rec["times_cue_sec"], dtype=float)
            smoothed = moving_average(probs, k=smooth_k)
            detected, first_detect_idx, _ = run_hysteresis_detector(
                smoothed_probs=smoothed,
                times_epoch_sec=times_epoch_sec,
                high_threshold=threshold,
                low_threshold=low_threshold,
                min_consecutive=min_consecutive,
                refractory_sec=refractory_sec,
                arm_after_cue_sec=arm_after_cue_sec,
            )
            if detected and first_detect_idx is not None:
                detection_latencies_epoch.append(float(times_epoch_sec[first_detect_idx]))
                detection_latencies_cue.append(float(times_cue_sec[first_detect_idx]))

    metrics = {
        **metric_core,
        "mi_hit_rate": hit_rate,
        "rest_false_alarm_rate": far,
        "mean_detection_latency_epoch_sec": float(np.mean(detection_latencies_epoch)) if detection_latencies_epoch else None,
        "mean_detection_latency_cue_sec": float(np.mean(detection_latencies_cue)) if detection_latencies_cue else None,
    }
    return metrics, trial_results



def compact_result_for_json(result_dict):
    out = dict(result_dict)
    out.pop("trial_results", None)
    out.pop("base_traces", None)
    return out



def metric_mean_std(items, key_path):
    vals = []
    for item in items:
        d = item
        for k in key_path:
            d = d[k]
        if d is not None:
            vals.append(float(d))
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=0)), "n": int(len(vals))}



def config_to_string(d):
    return (
        f"window=({d['train_epoch_tmin']},{d['train_epoch_tmax']})|"
        f"win={d['win_sec']}|step={d['step_sec']}|smooth_k={d['smooth_k']}|"
        f"thr={d['threshold']}|low={d['low_threshold']}|min_consec={d['min_consecutive_windows']}|"
        f"refrac={d['refractory_sec']}|arm={d['arm_after_cue_sec']}"
    )



def unconstrained_sort_key(metrics):
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



def aggregate_inner_results(inner_results):
    grouped = defaultdict(list)
    key_fields = [
        "train_epoch_tmin", "train_epoch_tmax", "win_sec", "step_sec",
        "smooth_k", "threshold", "low_threshold", "min_consecutive_windows",
        "refractory_sec", "arm_after_cue_sec",
    ]
    for r in inner_results:
        key = tuple(r[k] for k in key_fields)
        grouped[key].append(r)

    aggregated = []
    for _, rows in grouped.items():
        template = rows[0]
        pseudo_metrics_list = [r["pseudo_online_val_event_metrics"] for r in rows]
        epoch_metrics_list = [r["epoch_level_val_metrics"] for r in rows]

        def mean_metric(metric_name, metric_rows, default=None):
            vals = [m[metric_name] for m in metric_rows if m[metric_name] is not None]
            return float(np.mean(vals)) if vals else default

        agg_epoch = {
            "accuracy": mean_metric("accuracy", epoch_metrics_list),
            "balanced_accuracy": mean_metric("balanced_accuracy", epoch_metrics_list),
            "f1": mean_metric("f1", epoch_metrics_list),
            "n_inner_folds": int(len(rows)),
        }
        agg_pseudo = {
            "accuracy": mean_metric("accuracy", pseudo_metrics_list),
            "balanced_accuracy": mean_metric("balanced_accuracy", pseudo_metrics_list),
            "f1": mean_metric("f1", pseudo_metrics_list),
            "mi_hit_rate": mean_metric("mi_hit_rate", pseudo_metrics_list),
            "rest_false_alarm_rate": mean_metric("rest_false_alarm_rate", pseudo_metrics_list),
            "mean_detection_latency_epoch_sec": mean_metric("mean_detection_latency_epoch_sec", pseudo_metrics_list),
            "mean_detection_latency_cue_sec": mean_metric("mean_detection_latency_cue_sec", pseudo_metrics_list),
            "n_inner_folds": int(len(rows)),
        }
        aggregated.append(
            {
                "train_epoch_tmin": template["train_epoch_tmin"],
                "train_epoch_tmax": template["train_epoch_tmax"],
                "win_sec": template["win_sec"],
                "step_sec": template["step_sec"],
                "smooth_k": template["smooth_k"],
                "threshold": template["threshold"],
                "low_threshold": template["low_threshold"],
                "min_consecutive_windows": template["min_consecutive_windows"],
                "refractory_sec": template["refractory_sec"],
                "arm_after_cue_sec": template["arm_after_cue_sec"],
                "epoch_level_inner_val_metrics": agg_epoch,
                "pseudo_online_inner_val_event_metrics": agg_pseudo,
            }
        )

    aggregated.sort(
        key=lambda d: unconstrained_sort_key(d["pseudo_online_inner_val_event_metrics"]),
        reverse=True,
    )
    return aggregated



def concat_runs(epoch_cache, runs_to_use, tmin, tmax):
    xs, ys = [], []
    sfreq_ref = None
    for run_id in runs_to_use:
        X_run, y_run, sfreq = epoch_cache[(run_id, tmin, tmax)]
        xs.append(X_run)
        ys.append(y_run)
        if sfreq_ref is None:
            sfreq_ref = sfreq
        elif not np.isclose(sfreq_ref, sfreq):
            raise RuntimeError("Sampling frequency mismatch across runs")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), float(sfreq_ref)


# =========================
# Main subject routine
# =========================
def run_one_subject(data_dir, subject_id):
    print("=" * 80)
    print(f"Pseudo-online LORO with hysteresis/refractory | subject {subject_id}")
    print(f"RUNS           = {RUNS}")
    print(f"BAND           = {L_FREQ}-{H_FREQ} Hz")
    print(f"TRAIN_WINDOWS  = {TRAIN_WINDOWS}")
    print(f"WIN_SEC_LIST   = {WIN_SEC_LIST}")
    print(f"STEP_SEC_LIST  = {STEP_SEC_LIST}")
    print(f"SMOOTH_K_LIST  = {SMOOTH_K_LIST}")
    print(f"THRESHOLDS     = {THRESHOLD_LIST}")
    print(f"LOW_THRESHOLDS = {LOW_THRESHOLD_LIST}")
    print(f"MIN_CONSEC     = {MIN_CONSECUTIVE_LIST}")
    print(f"REFRACTORY     = {REFRACTORY_SEC_LIST}")
    print(f"ARM_AFTER_CUE  = {ARM_AFTER_CUE_SEC_LIST}")
    print(f"TIME_REFERENCE = {WINDOW_TIME_REFERENCE}")
    print("=" * 80)

    epoch_cache = {}
    run_trial_counts = {}
    run_label_counts = {}
    sfreq_ref = None

    for run_id in RUNS:
        edf_path = load_run_path(data_dir, subject_id, run_id)
        raw = load_run_raw(edf_path)
        raw_filt = raw.copy().filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="ERROR")

        ref_tmin = min(t[0] for t in TRAIN_WINDOWS)
        ref_tmax = max(t[1] for t in TRAIN_WINDOWS)
        X_ref, y_ref, sfreq = make_epochs_from_filtered_raw(raw_filt, ref_tmin, ref_tmax)
        run_trial_counts[run_id] = int(len(y_ref))
        run_label_counts[run_id] = np.bincount(y_ref).tolist()
        if sfreq_ref is None:
            sfreq_ref = sfreq

        for tmin, tmax in TRAIN_WINDOWS:
            X, y, sfreq2 = make_epochs_from_filtered_raw(raw_filt, tmin, tmax)
            if not np.array_equal(y, y_ref):
                raise RuntimeError(f"Run {run_id}: label order mismatch between windows")
            epoch_cache[(run_id, tmin, tmax)] = (X, y, sfreq2)

    fold_summaries = []

    for test_run in RUNS:
        dev_runs = [r for r in RUNS if r != test_run]
        print("-" * 80)
        print(f"Subject {subject_id} | test_run={test_run} | dev_runs={dev_runs}")

        inner_results = []
        for val_run in dev_runs:
            train_runs = [r for r in dev_runs if r != val_run]
            assert len(train_runs) == 1
            train_run = train_runs[0]

            for tmin, tmax in TRAIN_WINDOWS:
                X_train, y_train, sfreq = concat_runs(epoch_cache, [train_run], tmin, tmax)
                X_val, y_val, _ = concat_runs(epoch_cache, [val_run], tmin, tmax)

                clf = build_classifier(random_state=42 + subject_id * 10 + test_run * 100 + val_run)
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
                    for smooth_k, threshold, low_threshold, min_consecutive, refractory_sec, arm_after_cue_sec in itertools.product(
                        SMOOTH_K_LIST,
                        THRESHOLD_LIST,
                        LOW_THRESHOLD_LIST,
                        MIN_CONSECUTIVE_LIST,
                        REFRACTORY_SEC_LIST,
                        ARM_AFTER_CUE_SEC_LIST,
                    ):
                        if low_threshold > threshold:
                            continue
                        event_metrics, _ = evaluate_decision_layer_from_base_traces(
                            base_traces=base_traces,
                            smooth_k=smooth_k,
                            threshold=threshold,
                            low_threshold=low_threshold,
                            min_consecutive=min_consecutive,
                            refractory_sec=refractory_sec,
                            arm_after_cue_sec=arm_after_cue_sec,
                            include_trial_traces=False,
                        )
                        inner_results.append(
                            {
                                "subject_id": int(subject_id),
                                "test_run": int(test_run),
                                "inner_train_run": int(train_run),
                                "inner_val_run": int(val_run),
                                "train_epoch_tmin": float(tmin),
                                "train_epoch_tmax": float(tmax),
                                "win_sec": float(win_sec),
                                "step_sec": float(step_sec),
                                "smooth_k": int(smooth_k),
                                "threshold": float(threshold),
                                "low_threshold": float(low_threshold),
                                "min_consecutive_windows": int(min_consecutive),
                                "refractory_sec": float(refractory_sec),
                                "arm_after_cue_sec": float(arm_after_cue_sec),
                                "epoch_level_val_metrics": epoch_val_metrics,
                                "pseudo_online_val_event_metrics": event_metrics,
                            }
                        )

        aggregated_inner = aggregate_inner_results(inner_results)
        best_cfg = aggregated_inner[0]

        best_tmin = best_cfg["train_epoch_tmin"]
        best_tmax = best_cfg["train_epoch_tmax"]
        best_win_sec = best_cfg["win_sec"]
        best_step_sec = best_cfg["step_sec"]
        best_smooth_k = best_cfg["smooth_k"]
        best_threshold = best_cfg["threshold"]
        best_low_threshold = best_cfg["low_threshold"]
        best_min_consecutive = best_cfg["min_consecutive_windows"]
        best_refractory_sec = best_cfg["refractory_sec"]
        best_arm_after_cue_sec = best_cfg["arm_after_cue_sec"]

        X_dev, y_dev, sfreq_dev = concat_runs(epoch_cache, dev_runs, best_tmin, best_tmax)
        X_test, y_test, _ = concat_runs(epoch_cache, [test_run], best_tmin, best_tmax)

        clf_final = build_classifier(random_state=42 + subject_id * 10 + test_run)
        clf_final.fit(X_dev, y_dev)

        y_test_pred_epoch = clf_final.predict(X_test)
        epoch_test_metrics = compute_basic_metrics(y_test, y_test_pred_epoch)

        test_base_traces = replay_trials_collect_base_traces(
            clf=clf_final,
            X_epochs=X_test,
            y_epochs=y_test,
            sfreq=sfreq_dev,
            tmin=best_tmin,
            win_sec=best_win_sec,
            step_sec=best_step_sec,
            time_reference=WINDOW_TIME_REFERENCE,
        )
        pseudo_online_test_metrics, test_trial_results = evaluate_decision_layer_from_base_traces(
            base_traces=test_base_traces,
            smooth_k=best_smooth_k,
            threshold=best_threshold,
            low_threshold=best_low_threshold,
            min_consecutive=best_min_consecutive,
            refractory_sec=best_refractory_sec,
            arm_after_cue_sec=best_arm_after_cue_sec,
            include_trial_traces=SAVE_TEST_TRIAL_TRACES,
        )

        fold_summary = {
            "test_run": int(test_run),
            "dev_runs": [int(r) for r in dev_runs],
            "selection_status": "loro_inner_run_validation_hysteresis_refractory",
            "best_validation_config": {
                "train_epoch_tmin": float(best_tmin),
                "train_epoch_tmax": float(best_tmax),
                "win_sec": float(best_win_sec),
                "step_sec": float(best_step_sec),
                "smooth_k": int(best_smooth_k),
                "threshold": float(best_threshold),
                "low_threshold": float(best_low_threshold),
                "min_consecutive_windows": int(best_min_consecutive),
                "refractory_sec": float(best_refractory_sec),
                "arm_after_cue_sec": float(best_arm_after_cue_sec),
                "epoch_level_inner_val_metrics": best_cfg["epoch_level_inner_val_metrics"],
                "pseudo_online_inner_val_event_metrics": best_cfg["pseudo_online_inner_val_event_metrics"],
            },
            "held_out_test_epoch_level_metrics": epoch_test_metrics,
            "held_out_test_pseudo_online_event_level_metrics": pseudo_online_test_metrics,
            "top_inner_validation_results": [compact_result_for_json(r) for r in aggregated_inner[:NUM_TOP_INNER_VAL_RESULTS_TO_SAVE]],
        }
        if SAVE_TEST_TRIAL_TRACES:
            fold_summary["held_out_test_trial_traces"] = test_trial_results
        fold_summaries.append(fold_summary)

        print(
            f"test_run={test_run}: event_acc={pseudo_online_test_metrics['accuracy']:.4f}, "
            f"hit={pseudo_online_test_metrics['mi_hit_rate']}, "
            f"far={pseudo_online_test_metrics['rest_false_alarm_rate']}, "
            f"lat={pseudo_online_test_metrics['mean_detection_latency_cue_sec']}"
        )

    best_config_counter = Counter(config_to_string(f["best_validation_config"]) for f in fold_summaries)

    aggregate_test_metrics = {
        "epoch_accuracy": metric_mean_std(fold_summaries, ["held_out_test_epoch_level_metrics", "accuracy"]),
        "epoch_balanced_accuracy": metric_mean_std(fold_summaries, ["held_out_test_epoch_level_metrics", "balanced_accuracy"]),
        "epoch_f1": metric_mean_std(fold_summaries, ["held_out_test_epoch_level_metrics", "f1"]),
        "event_accuracy": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "accuracy"]),
        "event_balanced_accuracy": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "balanced_accuracy"]),
        "event_f1": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "f1"]),
        "hit_rate": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "mi_hit_rate"]),
        "false_alarm_rate": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "rest_false_alarm_rate"]),
        "detection_latency_epoch_sec": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_epoch_sec"]),
        "detection_latency_cue_sec": metric_mean_std(fold_summaries, ["held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_cue_sec"]),
    }

    subject_summary = {
        "subject_id": int(subject_id),
        "runs": RUNS,
        "l_freq": float(L_FREQ),
        "h_freq": float(H_FREQ),
        "time_reference": WINDOW_TIME_REFERENCE,
        "run_trial_counts": {str(k): v for k, v in run_trial_counts.items()},
        "run_label_counts": {str(k): v for k, v in run_label_counts.items()},
        "sampling_frequency": float(sfreq_ref),
        "num_outer_folds": int(len(RUNS)),
        "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "low_threshold_list": LOW_THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
            "refractory_sec_list": REFRACTORY_SEC_LIST,
            "arm_after_cue_sec_list": ARM_AFTER_CUE_SEC_LIST,
        },
        "aggregate_test_metrics_across_folds": aggregate_test_metrics,
        "best_config_frequency": [
            {"config": cfg, "count": int(cnt)} for cfg, cnt in best_config_counter.most_common()
        ],
        "fold_results": fold_summaries,
    }
    return subject_summary


# =========================
# Main
# =========================
def main():
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_results = []
    csv_rows = []

    for subject_id in SUBJECT_IDS:
        subject_summary = run_one_subject(data_dir, subject_id)
        subject_results.append(subject_summary)

        subj_json = output_dir / f"pseudo_online_loro_hysteresis_refractory_subject{subject_id}.json"
        with open(subj_json, "w", encoding="utf-8") as f:
            json.dump(subject_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved subject summary to: {subj_json}")

        for fold in subject_summary["fold_results"]:
            best = fold["best_validation_config"]
            testm = fold["held_out_test_pseudo_online_event_level_metrics"]
            csv_rows.append(
                {
                    "subject_id": subject_id,
                    "test_run": fold["test_run"],
                    "dev_runs": ",".join(map(str, fold["dev_runs"])),
                    "train_epoch_tmin": best["train_epoch_tmin"],
                    "train_epoch_tmax": best["train_epoch_tmax"],
                    "win_sec": best["win_sec"],
                    "step_sec": best["step_sec"],
                    "smooth_k": best["smooth_k"],
                    "threshold": best["threshold"],
                    "low_threshold": best["low_threshold"],
                    "min_consecutive_windows": best["min_consecutive_windows"],
                    "refractory_sec": best["refractory_sec"],
                    "arm_after_cue_sec": best["arm_after_cue_sec"],
                    "inner_val_event_bacc": best["pseudo_online_inner_val_event_metrics"]["balanced_accuracy"],
                    "inner_val_hit_rate": best["pseudo_online_inner_val_event_metrics"]["mi_hit_rate"],
                    "inner_val_far": best["pseudo_online_inner_val_event_metrics"]["rest_false_alarm_rate"],
                    "inner_val_latency_cue_sec": best["pseudo_online_inner_val_event_metrics"]["mean_detection_latency_cue_sec"],
                    "test_event_accuracy": testm["accuracy"],
                    "test_hit_rate": testm["mi_hit_rate"],
                    "test_far": testm["rest_false_alarm_rate"],
                    "test_latency_cue_sec": testm["mean_detection_latency_cue_sec"],
                }
            )

    overall = {
        "model": "PseudoOnline_Riemannian_TangentSpace_LR_MIvsRest_LORO_HysteresisRefractory",
        "subject_ids": SUBJECT_IDS,
        "num_subjects": len(SUBJECT_IDS),
        "runs": RUNS,
        "l_freq": float(L_FREQ),
        "h_freq": float(H_FREQ),
        "num_outer_folds_per_subject": len(RUNS),
        "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "low_threshold_list": LOW_THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
            "refractory_sec_list": REFRACTORY_SEC_LIST,
            "arm_after_cue_sec_list": ARM_AFTER_CUE_SEC_LIST,
        },
        "time_reference": WINDOW_TIME_REFERENCE,
        "subject_results": subject_results,
    }

    def mean_of_subject_means(metric_name):
        vals = []
        for subj in subject_results:
            v = subj["aggregate_test_metrics_across_folds"][metric_name]["mean"]
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    overall["overall_mean_of_subject_means"] = {
        "event_accuracy": mean_of_subject_means("event_accuracy"),
        "hit_rate": mean_of_subject_means("hit_rate"),
        "false_alarm_rate": mean_of_subject_means("false_alarm_rate"),
        "detection_latency_cue_sec": mean_of_subject_means("detection_latency_cue_sec"),
    }

    out_json = output_dir / "pseudo_online_loro_hysteresis_refractory_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    out_csv = output_dir / "pseudo_online_loro_hysteresis_refractory_selected_configs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("=" * 80)
    print("Finished all subjects")
    print(f"Overall event accuracy mean: {overall['overall_mean_of_subject_means']['event_accuracy']}")
    print(f"Overall hit rate mean: {overall['overall_mean_of_subject_means']['hit_rate']}")
    print(f"Overall false alarm rate mean: {overall['overall_mean_of_subject_means']['false_alarm_rate']}")
    print(f"Overall latency cue mean: {overall['overall_mean_of_subject_means']['detection_latency_cue_sec']}")
    print(f"Saved combined summary to: {out_json}")
    print(f"Saved selected config CSV to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
