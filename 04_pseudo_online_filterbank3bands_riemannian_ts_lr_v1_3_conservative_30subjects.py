#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-online MI-vs-Rest detection with 3-band Filter-bank Riemannian Tangent Space + Logistic Regression, conservative decision-layer v1.3.

Purpose
-------
This script is a 3-band v1.3 conservative revision of the filter-bank Riemannian benchmark.
It keeps the same model, data, and evaluation logic as the 3-band v1.2 script,
but uses a more conservative pseudo-online decision search space to reduce false alarms:

- subjects 1-10
- EEGBCI runs 4 / 8 / 12
- leave-one-run-out outer evaluation
- inner run validation for decision-layer selection
- training epoch window 0.5-3.5 s
- pseudo-online sliding-window replay
- smoothing + dual-threshold hysteresis + min-consecutive decision layer
- conservative decision candidates: stronger smoothing, high thresholds, min-consecutive fixed at 4, delayed arming
- selection targets: inner-val FAR <= 0.15, FAR std <= 0.10, hit >= 0.50

Main difference from the single-band Riemannian baseline
-------------------------------------------------------
Instead of using one broad 8-30 Hz covariance representation, this script uses a
filter bank:
    8-13, 13-20, 20-30 Hz
For each band:
    bandpass -> covariance -> tangent-space features
Then tangent-space features are concatenated across bands and classified with:
    StandardScaler -> LogisticRegression(class_weight='balanced')
Probability output is obtained with a cross-validated sigmoid calibrator fitted on
out-of-fold training scores.

Notes
-----
- The classifier is trained on the full training window, e.g. 0.5-3.5 s.
- During pseudo-online replay, the same classifier is applied to shorter sliding windows.
- This is intentional and follows a covariance-based pseudo-online usage pattern because
  covariance/tangent-space feature dimensionality does not depend on the time-window length.

April 29, 2026, XYan
Revision v1.3 3-band conservative: model unchanged; conservative decision-layer search space for stronger FAR control.
"""

from pathlib import Path
import json
import csv
import itertools
import random
from collections import Counter, defaultdict

import numpy as np
import mne

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace


# =============================================================================
# User settings
# =============================================================================

# Windows defaults. For Mac, change these paths accordingly.
DATA_DIR = r"D:\BCI\EEGBCI_Data"
OUTPUT_DIR = r"D:\BCI\EEGBCI_Results"

SUBJECT_IDS = list(range(1, 31))
RUNS = [4, 8, 12]

# 3-band filter-bank definition.
# Keeps the overall 8-30 Hz range but uses broader alpha / low-beta / high-beta bands.
BANDS = [
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
]

TRAIN_WINDOWS = [
    (0.5, 3.5),
]

WIN_SEC_LIST = [0.5, 0.75, 1.0]
STEP_SEC_LIST = [0.1, 0.125, 0.25]

# v1.3 conservative decision-layer search.
# Goal: reduce held-out false alarms by using stronger smoothing, higher thresholds,
# fixed min_consecutive=4, and delayed arming. Model and data settings are unchanged.
# Goal: reduce held-out false alarms while preserving the v1 hit-rate/latency advantage.
# Compared with v1, this removes very permissive candidates such as low thresholds,
# min_consecutive=1/2, smooth_k=1/3, and arm_after_cue_sec=0.0.
SMOOTH_K_LIST = [5, 7]

THRESHOLD_LIST = [0.8, 0.85, 0.9]
LOW_THRESHOLD_LIST = [0.7]
MIN_CONSECUTIVE_LIST = [4]

FIXED_REFRACTORY_SEC = 0.0
ARM_AFTER_CUE_SEC_LIST = [1.25, 1.5]

TARGET_MAX_INNER_VAL_FAR = 0.15
TARGET_MAX_INNER_VAL_FAR_STD = 0.10
TARGET_MIN_INNER_VAL_HIT = 0.50

WINDOW_TIME_REFERENCE = "end"
NUM_TOP_INNER_VAL_RESULTS_TO_SAVE = 10
SAVE_TEST_TRIAL_TRACES = False

# Riemannian model settings
COV_ESTIMATOR = "oas"
TANGENT_METRIC = "riemann"
LR_C = 1.0
LR_MAX_ITER = 2000
USE_SIGMOID_CALIBRATION = True
CALIBRATION_CV = 3

# MNE returns volts. Scaling to microvolts improves numerical conditioning for covariance.
EEG_SCALE = 1e6

RANDOM_SEED = 42

# Set True for a quick first test on subject 1, test run 4.
# After debug succeeds, set to False for the full 10-subject benchmark.
DEBUG_SINGLE_FOLD = False
DEBUG_SUBJECT_ID = 1
DEBUG_TEST_RUN = 4


# =============================================================================
# Reproducibility helpers
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# Data loading and epoching
# =============================================================================

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

    # EEGBCI annotations after MNE standardization are usually:
    # T0 -> rest, T1/T2 -> left/right imagery or movement depending on run.
    # For MI-vs-rest, T1 and T2 are merged into class 1.
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

    X = epochs.get_data().astype(np.float64) * EEG_SCALE
    event_codes = epochs.events[:, 2].astype(np.int64)
    y = np.where(event_codes == 1, 0, 1).astype(np.int64)
    return X, y, float(epochs.info["sfreq"])


# =============================================================================
# Sliding windows and trial replay
# =============================================================================

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
        windows = np.stack(windows, axis=0).astype(np.float64)
    else:
        windows = np.empty((0, n_channels, win_samples), dtype=np.float64)

    return windows, np.asarray(times_epoch_sec, dtype=float)


# =============================================================================
# Filter-bank Riemannian Tangent Space + Logistic Regression
# =============================================================================

class FilterBankRiemannianTSLR:
    """Filter-bank covariance -> tangent-space features -> LR + optional sigmoid calibration."""

    def __init__(
        self,
        bands,
        cov_estimator="oas",
        tangent_metric="riemann",
        lr_c=1.0,
        lr_max_iter=2000,
        use_sigmoid_calibration=True,
        calibration_cv=3,
        random_state=42,
    ):
        self.bands = list(bands)
        self.cov_estimator = cov_estimator
        self.tangent_metric = tangent_metric
        self.lr_c = float(lr_c)
        self.lr_max_iter = int(lr_max_iter)
        self.use_sigmoid_calibration = bool(use_sigmoid_calibration)
        self.calibration_cv = int(calibration_cv)
        self.random_state = int(random_state)

        self.band_covs_ = []
        self.band_tss_ = []
        self.scaler_ = None
        self.clf_ = None
        self.calibrator_ = None
        self.classes_ = None
        self.training_summary_ = {}

    def _new_lr(self):
        return LogisticRegression(
            C=self.lr_c,
            max_iter=self.lr_max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        )

    def _fit_feature_extractor(self, X_by_band, y=None):
        covs = []
        tss = []
        feats = []
        for Xb in X_by_band:
            cov = Covariances(estimator=self.cov_estimator)
            Cb = cov.fit_transform(Xb)
            ts = TangentSpace(metric=self.tangent_metric)
            Fb = ts.fit_transform(Cb)
            covs.append(cov)
            tss.append(ts)
            feats.append(Fb)
        return covs, tss, np.concatenate(feats, axis=1)

    @staticmethod
    def _transform_feature_extractor(X_by_band, covs, tss):
        feats = []
        for Xb, cov, ts in zip(X_by_band, covs, tss):
            Cb = cov.transform(Xb)
            Fb = ts.transform(Cb)
            feats.append(Fb)
        return np.concatenate(feats, axis=1)

    def _fit_base_on_features(self, F, y):
        scaler = StandardScaler()
        Fz = scaler.fit_transform(F)
        clf = self._new_lr()
        clf.fit(Fz, y)
        return scaler, clf

    def _raw_scores_from_model(self, X_by_band, covs, tss, scaler, clf):
        F = self._transform_feature_extractor(X_by_band, covs, tss)
        Fz = scaler.transform(F)
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(Fz)
            return np.asarray(scores, dtype=float).reshape(-1)
        probs = clf.predict_proba(Fz)[:, 1]
        eps = 1e-6
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.log(probs / (1.0 - probs))

    def _fit_oof_sigmoid_calibrator(self, X_by_band, y):
        y = np.asarray(y, dtype=int)
        min_class_count = int(np.min(np.bincount(y, minlength=2)))
        n_splits = min(int(self.calibration_cv), min_class_count)
        if n_splits < 2:
            return None, {
                "enabled": False,
                "method": "none",
                "reason": "Too few samples per class for calibration CV.",
            }

        oof_scores = np.zeros(len(y), dtype=float)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, cal_idx in cv.split(np.arange(len(y)), y):
            X_train_fold = [Xb[train_idx] for Xb in X_by_band]
            X_cal_fold = [Xb[cal_idx] for Xb in X_by_band]
            y_train_fold = y[train_idx]

            covs, tss, F_train = self._fit_feature_extractor(X_train_fold, y_train_fold)
            scaler, clf = self._fit_base_on_features(F_train, y_train_fold)
            oof_scores[cal_idx] = self._raw_scores_from_model(X_cal_fold, covs, tss, scaler, clf)

        calibrator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        calibrator.fit(oof_scores.reshape(-1, 1), y)

        cal_probs = calibrator.predict_proba(oof_scores.reshape(-1, 1))[:, 1]
        cal_pred = (cal_probs >= 0.5).astype(int)
        cal_meta = {
            "enabled": True,
            "method": "out_of_fold_sigmoid_logistic_regression",
            "input": "base_lr_decision_function",
            "cv": int(n_splits),
            "num_calibration_trials": int(len(y)),
            "calibration_class_counts_trials": np.bincount(y, minlength=2).astype(int).tolist(),
            "oof_score_mean": float(np.mean(oof_scores)),
            "oof_score_std": float(np.std(oof_scores, ddof=0)),
            "calibrated_mean_mi_prob_on_oof": float(np.mean(cal_probs)),
            "calibration_oof_accuracy": float(accuracy_score(y, cal_pred)),
            "calibration_oof_balanced_accuracy": float(balanced_accuracy_score(y, cal_pred)),
            "coef": calibrator.coef_.reshape(-1).astype(float).tolist(),
            "intercept": calibrator.intercept_.reshape(-1).astype(float).tolist(),
        }
        return calibrator, cal_meta

    def fit(self, X_by_band, y):
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This script expects exactly two classes: rest=0 and MI=1.")

        if self.use_sigmoid_calibration:
            calibrator, cal_meta = self._fit_oof_sigmoid_calibrator(X_by_band, y)
        else:
            calibrator = None
            cal_meta = {"enabled": False, "method": "none"}

        covs, tss, F = self._fit_feature_extractor(X_by_band, y)
        scaler, clf = self._fit_base_on_features(F, y)

        train_scores = self._raw_scores_from_model(X_by_band, covs, tss, scaler, clf)
        if calibrator is not None:
            train_probs = calibrator.predict_proba(train_scores.reshape(-1, 1))[:, 1]
        else:
            train_probs = clf.predict_proba(scaler.transform(F))[:, 1]
        train_pred = (train_probs >= 0.5).astype(int)

        self.band_covs_ = covs
        self.band_tss_ = tss
        self.scaler_ = scaler
        self.clf_ = clf
        self.calibrator_ = calibrator
        self.training_summary_ = {
            "num_train_trials": int(len(y)),
            "train_class_counts_trials": np.bincount(y, minlength=2).astype(int).tolist(),
            "num_bands": int(len(self.bands)),
            "bands": [[float(a), float(b)] for a, b in self.bands],
            "cov_estimator": self.cov_estimator,
            "tangent_metric": self.tangent_metric,
            "feature_dim": int(F.shape[1]),
            "lr_c": float(self.lr_c),
            "lr_max_iter": int(self.lr_max_iter),
            "train_probability_mean": float(np.mean(train_probs)),
            "train_epoch_accuracy_by_p05": float(accuracy_score(y, train_pred)),
            "train_epoch_balanced_accuracy_by_p05": float(balanced_accuracy_score(y, train_pred)),
            "probability_calibration": cal_meta,
        }
        return self

    def predict_proba(self, X_by_band):
        scores = self._raw_scores_from_model(
            X_by_band,
            self.band_covs_,
            self.band_tss_,
            self.scaler_,
            self.clf_,
        )
        if self.calibrator_ is not None:
            p1 = self.calibrator_.predict_proba(scores.reshape(-1, 1))[:, 1]
            p0 = 1.0 - p1
            return np.vstack([p0, p1]).T
        # Fallback to LR probability on final features.
        F = self._transform_feature_extractor(X_by_band, self.band_covs_, self.band_tss_)
        Fz = self.scaler_.transform(F)
        return self.clf_.predict_proba(Fz)


def fit_filterbank_riemannian_model(X_train_by_band, y_train, seed):
    model = FilterBankRiemannianTSLR(
        bands=BANDS,
        cov_estimator=COV_ESTIMATOR,
        tangent_metric=TANGENT_METRIC,
        lr_c=LR_C,
        lr_max_iter=LR_MAX_ITER,
        use_sigmoid_calibration=USE_SIGMOID_CALIBRATION,
        calibration_cv=CALIBRATION_CV,
        random_state=seed,
    )
    model.fit(X_train_by_band, y_train)
    return model


def replay_trials_collect_base_traces_filterbank(
    model,
    X_epochs_by_band,
    y_epochs,
    sfreq,
    tmin,
    win_sec,
    step_sec,
    time_reference,
):
    base_traces = []
    n_trials = len(y_epochs)
    n_bands = len(X_epochs_by_band)

    for i in range(n_trials):
        windows_by_band = []
        times_epoch_sec_ref = None
        for b in range(n_bands):
            windows, times_epoch_sec = sliding_windows_from_epoch(
                X_epochs_by_band[b][i],
                sfreq=sfreq,
                win_sec=win_sec,
                step_sec=step_sec,
                time_reference=time_reference,
            )
            windows_by_band.append(windows)
            if times_epoch_sec_ref is None:
                times_epoch_sec_ref = times_epoch_sec
            elif len(times_epoch_sec_ref) != len(times_epoch_sec) or not np.allclose(times_epoch_sec_ref, times_epoch_sec):
                raise RuntimeError("Sliding-window time mismatch across bands.")

        if len(windows_by_band[0]) == 0:
            continue

        probs = model.predict_proba(windows_by_band)[:, 1]
        base_traces.append(
            {
                "trial_index": int(i),
                "true_label": int(y_epochs[i]),
                "times_epoch_sec": times_epoch_sec_ref.astype(float).tolist(),
                "times_cue_sec": (times_epoch_sec_ref + tmin).astype(float).tolist(),
                "mi_probs": probs.astype(float).tolist(),
            }
        )
    return base_traces


def compute_epoch_average_metrics_from_base_traces(base_traces, threshold=0.5):
    y_true = []
    y_pred = []
    avg_probs = []
    for rec in base_traces:
        probs = np.asarray(rec["mi_probs"], dtype=float)
        if len(probs) == 0:
            continue
        p_mean = float(np.mean(probs))
        avg_probs.append(p_mean)
        y_true.append(int(rec["true_label"]))
        y_pred.append(1 if p_mean >= threshold else 0)
    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["mean_mi_probability"] = float(np.mean(avg_probs)) if avg_probs else None
    metrics["epoch_decision_rule"] = "mean_window_probability_ge_0.5"
    return metrics


# =============================================================================
# Decision layer and metrics
# =============================================================================

def moving_average(x, k=3):
    if k <= 1:
        return np.array(x, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - k + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def run_hysteresis_detector(
    smoothed_probs,
    times_cue_sec,
    high_threshold=0.6,
    low_threshold=0.5,
    min_consecutive=2,
    refractory_sec=0.0,
    arm_after_cue_sec=0.0,
):
    probs = np.asarray(smoothed_probs, dtype=float)
    times = np.asarray(times_cue_sec, dtype=float)

    if len(probs) != len(times):
        raise ValueError("smoothed_probs and times_cue_sec must have the same length")
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
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "confusion_matrix": cm.tolist(),
    }


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
            times_cue_sec=times_cue_sec,
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
                    "smoothed_mi_probs": smoothed.astype(float).tolist(),
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
        for tr in trial_results:
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
                times_cue_sec=times_cue_sec,
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


# =============================================================================
# Selection policy
# =============================================================================

def metric_mean_std(items, key_path):
    vals = []
    for item in items:
        x = item
        for k in key_path:
            x = x.get(k, None) if isinstance(x, dict) else None
            if x is None:
                break
        if x is not None:
            vals.append(float(x))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def _safe_balanced_metrics(metrics):
    far = metrics.get("rest_false_alarm_rate")
    far_std = metrics.get("rest_false_alarm_rate_std")
    hit = metrics.get("mi_hit_rate")
    hit_std = metrics.get("mi_hit_rate_std")
    lat = metrics.get("mean_detection_latency_cue_sec")
    bacc = metrics.get("balanced_accuracy")
    f1 = metrics.get("f1")

    far = 1.0 if far is None else float(far)
    far_std = 1.0 if far_std is None else float(far_std)
    hit = 0.0 if hit is None else float(hit)
    hit_std = 1.0 if hit_std is None else float(hit_std)
    lat = 99.0 if lat is None else float(lat)
    bacc = 0.0 if bacc is None else float(bacc)
    f1 = 0.0 if f1 is None else float(f1)
    return far, far_std, hit, hit_std, lat, bacc, f1


def joint_constraints_satisfied(metrics):
    far, far_std, hit, _, _, _, _ = _safe_balanced_metrics(metrics)
    return (far <= TARGET_MAX_INNER_VAL_FAR) and (far_std <= TARGET_MAX_INNER_VAL_FAR_STD) and (hit >= TARGET_MIN_INNER_VAL_HIT)


def far_constraints_satisfied(metrics):
    far, far_std, _, _, _, _, _ = _safe_balanced_metrics(metrics)
    return (far <= TARGET_MAX_INNER_VAL_FAR) and (far_std <= TARGET_MAX_INNER_VAL_FAR_STD)


def balanced_utility(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        2.5 * far
        + 1.0 * far_std
        + 1.0 * max(0.0, TARGET_MIN_INNER_VAL_HIT - hit)
        + 0.2 * hit_std
        + 0.15 * lat
        - 0.5 * bacc
        - 0.2 * f1
    )


def fallback_utility(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        2.0 * far
        + 0.7 * far_std
        + 1.5 * max(0.0, TARGET_MIN_INNER_VAL_HIT - hit)
        + 0.2 * lat
        - 0.4 * bacc
        - 0.2 * f1
    )


def balanced_robust_sort_key(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        balanced_utility(metrics),
        far,
        far_std,
        -hit,
        hit_std,
        lat,
        -bacc,
        -f1,
    )


def _joint_selection_sort_key(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        balanced_utility(metrics),
        far,
        far_std,
        -hit,
        hit_std,
        lat,
        -bacc,
        -f1,
    )


def _far_only_fallback_sort_key(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        max(0.0, TARGET_MIN_INNER_VAL_HIT - hit),
        balanced_utility(metrics),
        far_std,
        far,
        -hit,
        hit_std,
        -bacc,
        lat,
        -f1,
    )


def _global_fallback_sort_key(metrics):
    far, far_std, hit, hit_std, lat, bacc, f1 = _safe_balanced_metrics(metrics)
    return (
        fallback_utility(metrics),
        max(0.0, TARGET_MAX_INNER_VAL_FAR - far),
        max(0.0, TARGET_MIN_INNER_VAL_HIT - hit),
        far_std,
        far,
        -hit,
        hit_std,
        -bacc,
        lat,
        -f1,
    )


def select_best_aggregated_inner(aggregated_inner):
    joint = [d for d in aggregated_inner if joint_constraints_satisfied(d["pseudo_online_inner_val_event_metrics"])]
    far_only = [d for d in aggregated_inner if far_constraints_satisfied(d["pseudo_online_inner_val_event_metrics"])]

    selection_meta = {
        "num_joint_feasible_configs": int(len(joint)),
        "num_far_feasible_configs": int(len(far_only)),
        "selection_mode": None,
        "joint_constraints_satisfied": False,
        "far_constraints_satisfied": False,
        "fallback_reason": None,
    }

    if joint:
        joint_sorted = sorted(joint, key=lambda d: _joint_selection_sort_key(d["pseudo_online_inner_val_event_metrics"]))
        selection_meta.update(
            {
                "selection_mode": "joint_feasible",
                "joint_constraints_satisfied": True,
                "far_constraints_satisfied": True,
            }
        )
        return joint_sorted[0], selection_meta

    if far_only:
        far_sorted = sorted(far_only, key=lambda d: _far_only_fallback_sort_key(d["pseudo_online_inner_val_event_metrics"]))
        selection_meta.update(
            {
                "selection_mode": "fallback_far_feasible_only",
                "joint_constraints_satisfied": False,
                "far_constraints_satisfied": True,
                "fallback_reason": "No config satisfied FAR/FAR-std/hit jointly; selected from FAR-feasible configs only.",
            }
        )
        return far_sorted[0], selection_meta

    all_sorted = sorted(aggregated_inner, key=lambda d: _global_fallback_sort_key(d["pseudo_online_inner_val_event_metrics"]))
    selection_meta.update(
        {
            "selection_mode": "fallback_global_utility",
            "joint_constraints_satisfied": False,
            "far_constraints_satisfied": False,
            "fallback_reason": "No config satisfied FAR constraints; selected by global fallback utility.",
        }
    )
    return all_sorted[0], selection_meta


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

        def mean_and_std(metric_name, metric_rows):
            vals = [m[metric_name] for m in metric_rows if m.get(metric_name) is not None]
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals, ddof=0))

        epoch_acc_mean, epoch_acc_std = mean_and_std("accuracy", epoch_metrics_list)
        epoch_bacc_mean, epoch_bacc_std = mean_and_std("balanced_accuracy", epoch_metrics_list)
        epoch_f1_mean, epoch_f1_std = mean_and_std("f1", epoch_metrics_list)

        pseudo_acc_mean, pseudo_acc_std = mean_and_std("accuracy", pseudo_metrics_list)
        pseudo_bacc_mean, pseudo_bacc_std = mean_and_std("balanced_accuracy", pseudo_metrics_list)
        pseudo_f1_mean, pseudo_f1_std = mean_and_std("f1", pseudo_metrics_list)
        pseudo_hit_mean, pseudo_hit_std = mean_and_std("mi_hit_rate", pseudo_metrics_list)
        pseudo_far_mean, pseudo_far_std = mean_and_std("rest_false_alarm_rate", pseudo_metrics_list)
        pseudo_lat_epoch_mean, pseudo_lat_epoch_std = mean_and_std("mean_detection_latency_epoch_sec", pseudo_metrics_list)
        pseudo_lat_cue_mean, pseudo_lat_cue_std = mean_and_std("mean_detection_latency_cue_sec", pseudo_metrics_list)

        agg_epoch = {
            "accuracy": epoch_acc_mean,
            "accuracy_std": epoch_acc_std,
            "balanced_accuracy": epoch_bacc_mean,
            "balanced_accuracy_std": epoch_bacc_std,
            "f1": epoch_f1_mean,
            "f1_std": epoch_f1_std,
            "n_inner_folds": int(len(rows)),
            "epoch_decision_rule": "mean_window_probability_ge_0.5",
        }
        agg_pseudo = {
            "accuracy": pseudo_acc_mean,
            "accuracy_std": pseudo_acc_std,
            "balanced_accuracy": pseudo_bacc_mean,
            "balanced_accuracy_std": pseudo_bacc_std,
            "f1": pseudo_f1_mean,
            "f1_std": pseudo_f1_std,
            "mi_hit_rate": pseudo_hit_mean,
            "mi_hit_rate_std": pseudo_hit_std,
            "rest_false_alarm_rate": pseudo_far_mean,
            "rest_false_alarm_rate_std": pseudo_far_std,
            "mean_detection_latency_epoch_sec": pseudo_lat_epoch_mean,
            "mean_detection_latency_epoch_sec_std": pseudo_lat_epoch_std,
            "mean_detection_latency_cue_sec": pseudo_lat_cue_mean,
            "mean_detection_latency_cue_sec_std": pseudo_lat_cue_std,
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

    aggregated.sort(key=lambda d: balanced_robust_sort_key(d["pseudo_online_inner_val_event_metrics"]))
    return aggregated


# =============================================================================
# Utility functions
# =============================================================================

def concat_runs(epoch_cache, runs_to_use, tmin, tmax):
    """Return X_by_band list, y, sfreq for selected runs and training window."""
    X_by_band_accum = [[] for _ in BANDS]
    ys = []
    sfreq_ref = None

    for run_id in runs_to_use:
        for b_idx in range(len(BANDS)):
            X_run, y_run, sfreq = epoch_cache[(run_id, tmin, tmax, b_idx)]
            X_by_band_accum[b_idx].append(X_run)
            if b_idx == 0:
                ys.append(y_run)
                if sfreq_ref is None:
                    sfreq_ref = sfreq
                elif not np.isclose(sfreq_ref, sfreq):
                    raise RuntimeError("Sampling frequency mismatch across runs")

    X_by_band = [np.concatenate(parts, axis=0) for parts in X_by_band_accum]
    y = np.concatenate(ys, axis=0)
    return X_by_band, y, float(sfreq_ref)


def should_run_fold(subject_id, test_run):
    if not DEBUG_SINGLE_FOLD:
        return True
    return int(subject_id) == int(DEBUG_SUBJECT_ID) and int(test_run) == int(DEBUG_TEST_RUN)


def config_to_string(cfg):
    return (
        f"window=({cfg['train_epoch_tmin']},{cfg['train_epoch_tmax']})"
        f"|win={cfg['win_sec']}|step={cfg['step_sec']}|smooth_k={cfg['smooth_k']}"
        f"|thr={cfg['threshold']}|low={cfg['low_threshold']}"
        f"|min_consec={cfg['min_consecutive_windows']}|refrac={cfg['refractory_sec']}"
        f"|arm={cfg['arm_after_cue_sec']}"
    )


def aggregate_subject_fold_metrics(fold_summaries):
    metric_names = [
        "epoch_accuracy",
        "epoch_balanced_accuracy",
        "epoch_f1",
        "event_accuracy",
        "event_balanced_accuracy",
        "event_f1",
        "hit_rate",
        "false_alarm_rate",
        "detection_latency_epoch_sec",
        "detection_latency_cue_sec",
    ]
    out = {}
    mapping = {
        "epoch_accuracy": ("held_out_test_epoch_level_metrics", "accuracy"),
        "epoch_balanced_accuracy": ("held_out_test_epoch_level_metrics", "balanced_accuracy"),
        "epoch_f1": ("held_out_test_epoch_level_metrics", "f1"),
        "event_accuracy": ("held_out_test_pseudo_online_event_level_metrics", "accuracy"),
        "event_balanced_accuracy": ("held_out_test_pseudo_online_event_level_metrics", "balanced_accuracy"),
        "event_f1": ("held_out_test_pseudo_online_event_level_metrics", "f1"),
        "hit_rate": ("held_out_test_pseudo_online_event_level_metrics", "mi_hit_rate"),
        "false_alarm_rate": ("held_out_test_pseudo_online_event_level_metrics", "rest_false_alarm_rate"),
        "detection_latency_epoch_sec": ("held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_epoch_sec"),
        "detection_latency_cue_sec": ("held_out_test_pseudo_online_event_level_metrics", "mean_detection_latency_cue_sec"),
    }
    for name in metric_names:
        parent, child = mapping[name]
        vals = [f[parent].get(child) for f in fold_summaries if f[parent].get(child) is not None]
        out[name] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=0)) if vals else None,
            "n": int(len(vals)),
        }
    return out


# =============================================================================
# Main subject routine
# =============================================================================

def run_one_subject(data_dir, subject_id):
    print("=" * 80)
    print(f"Pseudo-online 3-band FilterBank Riemannian TS-LR v1.3 conservative | subject {subject_id}")
    print(f"RUNS               = {RUNS}")
    print(f"BANDS              = {BANDS}")
    print(f"TRAIN_WINDOWS      = {TRAIN_WINDOWS}")
    print(f"WIN_SEC_LIST       = {WIN_SEC_LIST}")
    print(f"STEP_SEC_LIST      = {STEP_SEC_LIST}")
    print(f"SMOOTH_K_LIST      = {SMOOTH_K_LIST}")
    print(f"THRESHOLDS         = {THRESHOLD_LIST}")
    print(f"LOW_THRESHOLDS     = {LOW_THRESHOLD_LIST}")
    print(f"MIN_CONSEC         = {MIN_CONSECUTIVE_LIST}")
    print(f"FIXED_REFRACTORY   = {FIXED_REFRACTORY_SEC}")
    print(f"ARM_AFTER_CUE      = {ARM_AFTER_CUE_SEC_LIST}")
    print(f"TARGET_MAX_FAR     = {TARGET_MAX_INNER_VAL_FAR}")
    print(f"TARGET_MAX_FAR_STD = {TARGET_MAX_INNER_VAL_FAR_STD}")
    print(f"TARGET_MIN_HIT     = {TARGET_MIN_INNER_VAL_HIT}")
    print(f"TIME_REFERENCE     = {WINDOW_TIME_REFERENCE}")
    print("=" * 80)

    fold_search_space = {
        "train_windows": TRAIN_WINDOWS,
        "win_sec_list": WIN_SEC_LIST,
        "step_sec_list": STEP_SEC_LIST,
        "smooth_k_list": SMOOTH_K_LIST,
        "threshold_list": THRESHOLD_LIST,
        "low_threshold_list": LOW_THRESHOLD_LIST,
        "min_consecutive_list": MIN_CONSECUTIVE_LIST,
        "arm_after_cue_sec_list": ARM_AFTER_CUE_SEC_LIST,
        "fixed_refractory_sec": FIXED_REFRACTORY_SEC,
    }

    epoch_cache = {}
    run_trial_counts = {}
    run_label_counts = {}
    sfreq_ref = None

    # Load each run once, then create band-specific filtered epochs.
    for run_id in RUNS:
        edf_path = load_run_path(data_dir, subject_id, run_id)
        raw = load_run_raw(edf_path)

        # Use the first band/window as label reference.
        ref_tmin = min(t[0] for t in TRAIN_WINDOWS)
        ref_tmax = max(t[1] for t in TRAIN_WINDOWS)
        y_ref = None

        for b_idx, (l_freq, h_freq) in enumerate(BANDS):
            print(f"Filtering subject={subject_id}, run={run_id}, band={l_freq}-{h_freq} Hz")
            raw_filt = raw.copy().filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
            for tmin, tmax in TRAIN_WINDOWS:
                X, y, sfreq = make_epochs_from_filtered_raw(raw_filt, tmin, tmax)
                if y_ref is None:
                    y_ref = y.copy()
                    run_trial_counts[run_id] = int(len(y))
                    run_label_counts[run_id] = np.bincount(y, minlength=2).astype(int).tolist()
                    if sfreq_ref is None:
                        sfreq_ref = sfreq
                else:
                    if not np.array_equal(y, y_ref):
                        raise RuntimeError(f"Run {run_id}: label order mismatch across bands/windows")
                epoch_cache[(run_id, tmin, tmax, b_idx)] = (X, y, sfreq)

    fold_summaries = []

    for test_run in RUNS:
        if not should_run_fold(subject_id, test_run):
            print(f"Skipping subject {subject_id}, test_run={test_run} due to DEBUG_SINGLE_FOLD.")
            continue

        dev_runs = [r for r in RUNS if r != test_run]
        print("-" * 80)
        print(f"Subject {subject_id} | test_run={test_run} | dev_runs={dev_runs}")

        inner_results = []
        inner_model_training_summaries = []

        for val_run in dev_runs:
            train_runs = [r for r in dev_runs if r != val_run]
            assert len(train_runs) == 1
            train_run = train_runs[0]

            for tmin, tmax in fold_search_space["train_windows"]:
                X_train_by_band, y_train, sfreq = concat_runs(epoch_cache, [train_run], tmin, tmax)
                X_val_by_band, y_val, _ = concat_runs(epoch_cache, [val_run], tmin, tmax)

                seed = RANDOM_SEED + subject_id * 1000 + test_run * 100 + val_run * 10
                print(
                    f"Training FilterBank Riemannian TS-LR | subj={subject_id} test_run={test_run} "
                    f"train_run={train_run} val_run={val_run} seed={seed}"
                )
                model = fit_filterbank_riemannian_model(X_train_by_band, y_train, seed=seed)
                inner_model_training_summaries.append(
                    {
                        "inner_train_run": int(train_run),
                        "inner_val_run": int(val_run),
                        "training_summary": model.training_summary_,
                    }
                )

                for win_sec in fold_search_space["win_sec_list"]:
                    for step_sec in fold_search_space["step_sec_list"]:
                        base_traces = replay_trials_collect_base_traces_filterbank(
                            model=model,
                            X_epochs_by_band=X_val_by_band,
                            y_epochs=y_val,
                            sfreq=sfreq,
                            tmin=tmin,
                            win_sec=win_sec,
                            step_sec=step_sec,
                            time_reference=WINDOW_TIME_REFERENCE,
                        )
                        epoch_val_metrics = compute_epoch_average_metrics_from_base_traces(base_traces, threshold=0.5)

                        for smooth_k, threshold, low_threshold, min_consecutive, arm_after_cue_sec in itertools.product(
                            fold_search_space["smooth_k_list"],
                            fold_search_space["threshold_list"],
                            fold_search_space["low_threshold_list"],
                            fold_search_space["min_consecutive_list"],
                            fold_search_space["arm_after_cue_sec_list"],
                        ):
                            if low_threshold > threshold:
                                continue
                            event_metrics, _ = evaluate_decision_layer_from_base_traces(
                                base_traces=base_traces,
                                smooth_k=smooth_k,
                                threshold=threshold,
                                low_threshold=low_threshold,
                                min_consecutive=min_consecutive,
                                refractory_sec=FIXED_REFRACTORY_SEC,
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
                                    "refractory_sec": float(FIXED_REFRACTORY_SEC),
                                    "arm_after_cue_sec": float(arm_after_cue_sec),
                                    "epoch_level_val_metrics": epoch_val_metrics,
                                    "pseudo_online_val_event_metrics": event_metrics,
                                }
                            )

        aggregated_inner = aggregate_inner_results(inner_results)
        best_cfg, selection_meta = select_best_aggregated_inner(aggregated_inner)

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

        X_dev_by_band, y_dev, sfreq_dev = concat_runs(epoch_cache, dev_runs, best_tmin, best_tmax)
        X_test_by_band, y_test, _ = concat_runs(epoch_cache, [test_run], best_tmin, best_tmax)

        final_seed = RANDOM_SEED + subject_id * 1000 + test_run * 100
        print(
            f"Training FINAL FilterBank Riemannian TS-LR | subj={subject_id} test_run={test_run} "
            f"dev_runs={dev_runs} seed={final_seed}"
        )
        final_model = fit_filterbank_riemannian_model(X_dev_by_band, y_dev, seed=final_seed)

        test_base_traces = replay_trials_collect_base_traces_filterbank(
            model=final_model,
            X_epochs_by_band=X_test_by_band,
            y_epochs=y_test,
            sfreq=sfreq_dev,
            tmin=best_tmin,
            win_sec=best_win_sec,
            step_sec=best_step_sec,
            time_reference=WINDOW_TIME_REFERENCE,
        )
        epoch_test_metrics = compute_epoch_average_metrics_from_base_traces(test_base_traces, threshold=0.5)
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
            "selection_status": f"loro_inner_run_validation_filterbank_riemannian_ts_lr_v1_3_conservative_3bands::{selection_meta['selection_mode']}",
            "best_validation_config": {
                "selection_target_max_far": float(TARGET_MAX_INNER_VAL_FAR),
                "selection_target_max_far_std": float(TARGET_MAX_INNER_VAL_FAR_STD),
                "selection_target_min_hit": float(TARGET_MIN_INNER_VAL_HIT),
                "base_model": "FilterBank_Riemannian_TangentSpace_LR",
                "probability_output": "cv_sigmoid_calibrated_lr_decision_function" if USE_SIGMOID_CALIBRATION else "lr_predict_proba_no_extra_calibration",
                "selection_mode": selection_meta["selection_mode"],
                "joint_constraints_satisfied": bool(selection_meta["joint_constraints_satisfied"]),
                "far_constraints_satisfied": bool(selection_meta["far_constraints_satisfied"]),
                "num_joint_feasible_configs": int(selection_meta["num_joint_feasible_configs"]),
                "num_far_feasible_configs": int(selection_meta["num_far_feasible_configs"]),
                "fallback_reason": selection_meta["fallback_reason"],
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
            "final_model_training_summary": final_model.training_summary_,
            "inner_model_training_summaries": inner_model_training_summaries,
            "held_out_test_epoch_level_metrics": epoch_test_metrics,
            "held_out_test_pseudo_online_event_level_metrics": pseudo_online_test_metrics,
            "top_inner_validation_results": aggregated_inner[:NUM_TOP_INNER_VAL_RESULTS_TO_SAVE],
        }
        if SAVE_TEST_TRIAL_TRACES:
            fold_summary["held_out_test_trial_traces"] = test_trial_results

        print("Held-out test epoch metrics:", epoch_test_metrics)
        print("Held-out test pseudo-online metrics:", pseudo_online_test_metrics)

        fold_summaries.append(fold_summary)

    if not fold_summaries:
        return None

    aggregate_test_metrics = aggregate_subject_fold_metrics(fold_summaries)
    best_config_counter = Counter(config_to_string(f["best_validation_config"]) for f in fold_summaries)

    subject_summary = {
        "subject_id": int(subject_id),
        "runs": [int(r) for r in RUNS],
        "bands": [[float(a), float(b)] for a, b in BANDS],
        "time_reference": WINDOW_TIME_REFERENCE,
        "run_trial_counts": {str(k): int(v) for k, v in run_trial_counts.items()},
        "run_label_counts": {str(k): v for k, v in run_label_counts.items()},
        "sampling_frequency": float(sfreq_ref),
        "num_outer_folds": int(len(fold_summaries)),
        "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "low_threshold_list": LOW_THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
            "fixed_refractory_sec": FIXED_REFRACTORY_SEC,
            "arm_after_cue_sec_list": ARM_AFTER_CUE_SEC_LIST,
            "target_max_inner_val_far": TARGET_MAX_INNER_VAL_FAR,
            "target_max_inner_val_far_std": TARGET_MAX_INNER_VAL_FAR_STD,
            "target_min_inner_val_hit": TARGET_MIN_INNER_VAL_HIT,
        },
        "model_settings": {
            "base_model": "FilterBank_Riemannian_TangentSpace_LR",
            "bands": [[float(a), float(b)] for a, b in BANDS],
            "cov_estimator": COV_ESTIMATOR,
            "tangent_metric": TANGENT_METRIC,
            "lr_c": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "probability_output": "cv_sigmoid_calibrated_lr_decision_function" if USE_SIGMOID_CALIBRATION else "lr_predict_proba_no_extra_calibration",
            "use_sigmoid_calibration": USE_SIGMOID_CALIBRATION,
            "calibration_cv": CALIBRATION_CV,
            "eeg_scale": EEG_SCALE,
        },
        "aggregate_test_metrics_across_folds": aggregate_test_metrics,
        "best_config_frequency": [
            {"config": cfg, "count": int(cnt)} for cfg, cnt in best_config_counter.most_common()
        ],
        "fold_results": fold_summaries,
    }
    return subject_summary


# =============================================================================
# Main
# =============================================================================

def main():
    set_global_seed(RANDOM_SEED)

    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_results = []
    csv_rows = []

    for subject_id in SUBJECT_IDS:
        if DEBUG_SINGLE_FOLD and subject_id != DEBUG_SUBJECT_ID:
            continue

        subject_summary = run_one_subject(data_dir, subject_id)
        if subject_summary is None:
            continue

        subject_results.append(subject_summary)

        subj_json = output_dir / f"pseudo_online_filterbank3bands_riemannian_ts_lr_v1_3_conservative_30subjects_subject{subject_id}.json"
        with open(subj_json, "w", encoding="utf-8") as f:
            json.dump(subject_summary, f, indent=2, ensure_ascii=False)
        print(f"Saved subject summary to: {subj_json}")

        for fold in subject_summary["fold_results"]:
            best = fold["best_validation_config"]
            inner = best["pseudo_online_inner_val_event_metrics"]
            testm = fold["held_out_test_pseudo_online_event_level_metrics"]
            final_train = fold["final_model_training_summary"]
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
                    "selection_mode": best["selection_mode"],
                    "joint_constraints_satisfied": best["joint_constraints_satisfied"],
                    "num_joint_feasible_configs": best["num_joint_feasible_configs"],
                    "selection_target_max_far": best["selection_target_max_far"],
                    "selection_target_max_far_std": best["selection_target_max_far_std"],
                    "selection_target_min_hit": best["selection_target_min_hit"],
                    "inner_val_event_bacc": inner["balanced_accuracy"],
                    "inner_val_event_bacc_std": inner["balanced_accuracy_std"],
                    "inner_val_hit_rate": inner["mi_hit_rate"],
                    "inner_val_hit_rate_std": inner["mi_hit_rate_std"],
                    "inner_val_far": inner["rest_false_alarm_rate"],
                    "inner_val_far_std": inner["rest_false_alarm_rate_std"],
                    "inner_val_latency_cue_sec": inner["mean_detection_latency_cue_sec"],
                    "inner_val_latency_cue_sec_std": inner["mean_detection_latency_cue_sec_std"],
                    "test_event_accuracy": testm["accuracy"],
                    "test_event_bacc": testm["balanced_accuracy"],
                    "test_event_f1": testm["f1"],
                    "test_hit_rate": testm["mi_hit_rate"],
                    "test_far": testm["rest_false_alarm_rate"],
                    "test_latency_cue_sec": testm["mean_detection_latency_cue_sec"],
                    "final_num_train_trials": final_train["num_train_trials"],
                    "final_train_bacc_by_p05": final_train["train_epoch_balanced_accuracy_by_p05"],
                    "final_feature_dim": final_train["feature_dim"],
                    "calibration_enabled": final_train["probability_calibration"].get("enabled"),
                    "calibration_method": final_train["probability_calibration"].get("method"),
                }
            )

    overall = {
        "model": "PseudoOnline_FilterBank3Bands_Riemannian_TangentSpace_LR_MIvsRest_LORO_BalancedRobustCalV1_3Conservative",
        "subject_ids": SUBJECT_IDS if not DEBUG_SINGLE_FOLD else [DEBUG_SUBJECT_ID],
        "num_subjects": len(subject_results),
        "runs": RUNS,
        "bands": [[float(a), float(b)] for a, b in BANDS],
        "num_outer_folds_per_subject": len(RUNS) if not DEBUG_SINGLE_FOLD else 1,
        "search_space": {
            "train_windows": [list(t) for t in TRAIN_WINDOWS],
            "win_sec_list": WIN_SEC_LIST,
            "step_sec_list": STEP_SEC_LIST,
            "smooth_k_list": SMOOTH_K_LIST,
            "threshold_list": THRESHOLD_LIST,
            "low_threshold_list": LOW_THRESHOLD_LIST,
            "min_consecutive_list": MIN_CONSECUTIVE_LIST,
            "fixed_refractory_sec": FIXED_REFRACTORY_SEC,
            "arm_after_cue_sec_list": ARM_AFTER_CUE_SEC_LIST,
            "target_max_inner_val_far": TARGET_MAX_INNER_VAL_FAR,
            "target_max_inner_val_far_std": TARGET_MAX_INNER_VAL_FAR_STD,
            "target_min_inner_val_hit": TARGET_MIN_INNER_VAL_HIT,
        },
        "model_settings": {
            "base_model": "FilterBank_Riemannian_TangentSpace_LR",
            "bands": [[float(a), float(b)] for a, b in BANDS],
            "cov_estimator": COV_ESTIMATOR,
            "tangent_metric": TANGENT_METRIC,
            "lr_c": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "probability_output": "cv_sigmoid_calibrated_lr_decision_function" if USE_SIGMOID_CALIBRATION else "lr_predict_proba_no_extra_calibration",
            "use_sigmoid_calibration": USE_SIGMOID_CALIBRATION,
            "calibration_cv": CALIBRATION_CV,
            "eeg_scale": EEG_SCALE,
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

    overall["selection_policy"] = {
        "strategy": "balanced_robust_filterbank3bands_riemannian_ts_lr_v1_3_conservative",
        "target_max_inner_val_far": TARGET_MAX_INNER_VAL_FAR,
        "target_max_inner_val_far_std": TARGET_MAX_INNER_VAL_FAR_STD,
        "target_min_inner_val_hit": TARGET_MIN_INNER_VAL_HIT,
        "fixed_refractory_sec": FIXED_REFRACTORY_SEC,
        "arm_after_cue_is_cue_referenced": True,
        "probability_calibration": {
            "method": "out_of_fold_sigmoid_logistic_regression" if USE_SIGMOID_CALIBRATION else "none",
            "calibration_data": "training trials only, with CV out-of-fold scores",
            "input": "base LR decision_function score",
            "note": "Thresholds are selected on inner validation; held-out test remains untouched.",
        },
    }

    overall["overall_mean_of_subject_means"] = {
        "event_accuracy": mean_of_subject_means("event_accuracy"),
        "event_balanced_accuracy": mean_of_subject_means("event_balanced_accuracy"),
        "event_f1": mean_of_subject_means("event_f1"),
        "hit_rate": mean_of_subject_means("hit_rate"),
        "false_alarm_rate": mean_of_subject_means("false_alarm_rate"),
        "detection_latency_cue_sec": mean_of_subject_means("detection_latency_cue_sec"),
    }

    suffix = "debug" if DEBUG_SINGLE_FOLD else "results"
    out_json = output_dir / f"pseudo_online_filterbank3bands_riemannian_ts_lr_v1_3_conservative_30subjects_{suffix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    out_csv = output_dir / f"pseudo_online_filterbank3bands_riemannian_ts_lr_v1_3_conservative_30subjects_{suffix}_selected_configs.csv"
    if csv_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print("=" * 80)
    print("Finished 3-band FilterBank Riemannian TS-LR conservative v1.3 benchmark")
    print(f"Overall event accuracy mean: {overall['overall_mean_of_subject_means']['event_accuracy']}")
    print(f"Overall event balanced accuracy mean: {overall['overall_mean_of_subject_means']['event_balanced_accuracy']}")
    print(f"Overall hit rate mean: {overall['overall_mean_of_subject_means']['hit_rate']}")
    print(f"Overall false alarm rate mean: {overall['overall_mean_of_subject_means']['false_alarm_rate']}")
    print(f"Overall latency cue mean: {overall['overall_mean_of_subject_means']['detection_latency_cue_sec']}")
    print(f"Saved combined summary to: {out_json}")
    print(f"Saved selected config CSV to: {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
