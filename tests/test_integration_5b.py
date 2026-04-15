"""Autograder tests for Integration 5B — Model Comparison & Decision Memo."""

import pytest
import sys
import os
import tempfile
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_comparison import (load_and_prepare, build_preprocessor,
                               define_models, evaluate_all,
                               save_results, log_experiment,
                               save_best_model,
                               plot_pr_curves, plot_calibration)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "telecom_churn.csv"
)


# ── Data Loading ──────────────────────────────────────────────────────────

def test_data_loaded():
    """load_and_prepare returns (X, y) with correct shape and no target leak."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None"
    X, y = result
    assert X.shape[0] > 1000, f"Expected >1000 rows, got {X.shape[0]}"
    assert "churned" not in X.columns, "Target should not be in features"
    assert len(y) == len(X), "X and y must have same length"
    assert set(y.unique()).issubset({0, 1}), "Target should be binary (0/1)"


def test_features_include_numeric_and_categorical():
    """Features should include both numeric and categorical columns."""
    X, _ = load_and_prepare(DATA_PATH)
    numeric_expected = {"tenure", "monthly_charges", "total_charges"}
    categorical_expected = {"contract_type", "internet_service"}
    assert numeric_expected.issubset(set(X.columns)), (
        f"Missing numeric features: {numeric_expected - set(X.columns)}"
    )
    assert categorical_expected.issubset(set(X.columns)), (
        f"Missing categorical features: {categorical_expected - set(X.columns)}"
    )


# ── Preprocessor ──────────────────────────────────────────────────────────

def test_preprocessor():
    """build_preprocessor returns a working ColumnTransformer."""
    prep = build_preprocessor()
    assert prep is not None, "build_preprocessor returned None"
    assert hasattr(prep, "fit_transform"), "Preprocessor must have fit_transform"


def test_preprocessor_transforms_data():
    """Preprocessor transforms data without error and expands columns."""
    X, _ = load_and_prepare(DATA_PATH)
    prep = build_preprocessor()
    transformed = prep.fit_transform(X)
    assert transformed is not None
    assert transformed.shape[0] == X.shape[0], "Row count must be preserved"
    assert transformed.shape[1] > X.shape[1], (
        "Transformed data should have more columns (OneHotEncoder expansion)"
    )


# ── Model Definitions ────────────────────────────────────────────────────

def test_models_defined():
    """define_models returns at least 6 named Pipelines."""
    models = define_models()
    assert models is not None, "define_models returned None"
    assert len(models) >= 6, f"Expected >= 6 models, got {len(models)}"
    for name, pipe in models.items():
        assert hasattr(pipe, "fit"), f"'{name}' must have fit method"
        assert hasattr(pipe, "predict"), f"'{name}' must have predict method"


def test_models_are_pipelines():
    """Each model should be a Pipeline (preprocessor + estimator)."""
    from sklearn.pipeline import Pipeline
    models = define_models()
    assert models is not None
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), (
            f"'{name}' should be a Pipeline, got {type(pipe).__name__}"
        )


def test_models_include_both_families():
    """Models must include both linear and tree-based model families."""
    models = define_models()
    assert models is not None
    names_lower = [n.lower() for n in models.keys()]
    has_linear = any("log" in n or "ridge" in n for n in names_lower)
    has_tree = any("tree" in n or "forest" in n or "rf" in n for n in names_lower)
    assert has_linear, "Models must include at least one linear model (LogReg or Ridge)"
    assert has_tree, "Models must include at least one tree-based model (DecisionTree or RandomForest)"


def test_models_include_dummy_baseline():
    """A DummyClassifier baseline must be among the defined models."""
    models = define_models()
    assert models is not None
    names_lower = [n.lower() for n in models.keys()]
    assert any("dummy" in n or "baseline" in n for n in names_lower), (
        "Models must include a DummyClassifier baseline"
    )


# ── Evaluation ────────────────────────────────────────────────────────────

def test_evaluation_runs():
    """evaluate_all returns a DataFrame with expected shape and columns."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    assert models is not None

    results = evaluate_all(models, X, y)
    assert results is not None, "evaluate_all returned None"
    assert isinstance(results, pd.DataFrame), "Results must be a DataFrame"
    assert len(results) >= 6, f"Expected >= 6 rows, got {len(results)}"


def test_evaluation_has_required_columns():
    """Results DataFrame must contain mean columns for all required metrics."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    results = evaluate_all(models, X, y)
    assert results is not None

    required_cols = [
        "accuracy_mean", "precision_mean", "recall_mean",
        "f1_mean", "pr_auc_mean",
    ]
    for col in required_cols:
        assert col in results.columns, f"Missing column: {col}"


def test_evaluation_metrics_are_reasonable():
    """Metric values should be in [0, 1] and real models should beat baseline."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    results = evaluate_all(models, X, y)
    assert results is not None

    for col in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]:
        if col in results.columns:
            values = results[col].values
            assert all(0 <= v <= 1 for v in values), (
                f"All {col} values should be in [0, 1]"
            )


# ── Output Files ──────────────────────────────────────────────────────────

def test_save_results_creates_csv():
    """save_results should create a comparison_table.csv file."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    results = evaluate_all(models, X, y)
    assert results is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_results(results, output_dir=tmpdir)
        csv_path = os.path.join(tmpdir, "comparison_table.csv")
        assert os.path.exists(csv_path), "comparison_table.csv not created"
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) >= 6, "CSV should contain all model results"


def test_log_experiment_creates_csv():
    """log_experiment should create an experiment_log.csv with timestamps."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    results = evaluate_all(models, X, y)
    assert results is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        log_experiment(results, output_dir=tmpdir)
        log_path = os.path.join(tmpdir, "experiment_log.csv")
        assert os.path.exists(log_path), "experiment_log.csv not created"
        log_df = pd.read_csv(log_path)
        assert len(log_df) >= 6, "Log should contain one row per model"
        assert "timestamp" in log_df.columns, "Log must include a timestamp column"


def test_save_best_model_creates_joblib():
    """save_best_model should save a .joblib file."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    results = evaluate_all(models, X, y)
    assert results is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_best_model(models, results, X, y, output_dir=tmpdir)
        joblib_path = os.path.join(tmpdir, "best_model.joblib")
        assert os.path.exists(joblib_path), "best_model.joblib not created"
        # Verify the saved model can be loaded
        from joblib import load
        loaded = load(joblib_path)
        assert hasattr(loaded, "predict"), "Loaded model must have predict method"


# ── Plot Files ───────────────────────────────────────────────────────────

def test_plot_pr_curves_creates_file():
    """plot_pr_curves should create a pr_curves.png in the output directory."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    assert models is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_pr_curves(models, X, y, output_dir=tmpdir)
        plot_path = os.path.join(tmpdir, "pr_curves.png")
        assert os.path.exists(plot_path), (
            "pr_curves.png not created — plot_pr_curves must honor output_dir "
            "and save the figure there"
        )


def test_plot_calibration_creates_file():
    """plot_calibration should create a calibration.png in the output directory."""
    X, y = load_and_prepare(DATA_PATH)
    models = define_models()
    assert models is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_calibration(models, X, y, output_dir=tmpdir)
        plot_path = os.path.join(tmpdir, "calibration.png")
        assert os.path.exists(plot_path), (
            "calibration.png not created — plot_calibration must honor output_dir "
            "and save the figure there"
        )
