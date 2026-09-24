"""Candidate CNN evaluation and deterministic trial metadata."""

from __future__ import annotations

import gc
import platform
import time
from typing import Any

import numpy as np

from cnn_model import build_model
from config import ExperimentConfig, mode_interpretation
from dataset import DatasetBundle
from evaluation_contract import EvaluationResult, candidate_id, canonical_json, trial_seed


__all__ = [
    "EvaluationResult",
    "candidate_id",
    "canonical_json",
    "trial_seed",
    "evaluate_candidate",
    "train_final_model",
]


def environment_metadata() -> dict[str, str]:
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import tensorflow as tf

        metadata["tensorflow_version"] = tf.__version__
        metadata["keras_version"] = str(tf.keras.__version__)
        devices = tf.config.list_physical_devices("GPU")
        metadata["gpu_available"] = str(bool(devices)).lower()
        metadata["gpu_devices"] = ";".join(device.name for device in devices)
        metadata["gpu_utilization"] = "not_sampled"
    except (ImportError, AttributeError):
        metadata["tensorflow_version"] = "unavailable"
        metadata["keras_version"] = "unavailable"
        metadata["gpu_available"] = "unknown"
        metadata["gpu_devices"] = "unknown"
        metadata["gpu_utilization"] = "not_sampled"
    return metadata


def seed_tensorflow(seed: int, deterministic: bool) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    except (AttributeError, RuntimeError):
        pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (AttributeError, RuntimeError):
        pass
    warning = ""
    effective = False
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            effective = True
        except (AttributeError, RuntimeError) as error:
            warning = str(error)
    return {
        "determinism_requested": deterministic,
        "determinism_effective": effective if deterministic else False,
        "determinism_warning": warning,
    }


def _history_best(history: dict[str, list[float]]) -> tuple[int, float, float]:
    accuracies = history.get(
        "val_sparse_categorical_accuracy",
        history.get("val_categorical_accuracy", history.get("val_accuracy", [])),
    )
    losses = history.get("val_loss", [])
    if not accuracies or len(losses) != len(accuracies):
        raise ValueError("Training history did not contain validation accuracy")
    best = min(
        range(len(accuracies)),
        key=lambda index: (-float(accuracies[index]), float(losses[index]), index),
    )
    return best + 1, float(accuracies[best]), float(losses[best])


def _training_accuracy(history: dict[str, list[float]], epoch: int) -> float:
    accuracies = history.get(
        "sparse_categorical_accuracy",
        history.get("categorical_accuracy", history.get("accuracy", [])),
    )
    if len(accuracies) < epoch:
        raise ValueError("Training history did not contain training accuracy")
    return float(accuracies[epoch - 1])


def _semantic_warning(configuration: dict[str, Any], mode: str) -> str:
    if mode in {"paper_literal", "paper_conventional"} and configuration.get(
        "loss", "sparse_categorical_crossentropy"
    ) != "sparse_categorical_crossentropy":
        return "nonstandard_loss_for_multiclass_sparse_labels"
    return ""


def _labels_for_loss(labels: np.ndarray, configuration: dict[str, Any], mode: str) -> np.ndarray:
    """Adapt integer MNIST labels for paper losses that require class vectors."""
    loss = (
        "sparse_categorical_crossentropy"
        if mode == "improved"
        else configuration.get("loss", "sparse_categorical_crossentropy")
    )
    if loss == "sparse_categorical_crossentropy":
        return labels
    return np.eye(10, dtype=np.float32)[np.asarray(labels, dtype=np.int64)]


def _training_batch_size(configuration: dict[str, Any]) -> int:
    """Keep large paper candidates within the available laptop-GPU memory."""
    return min(int(configuration["batch_size"]), 32)


def _evaluation_metadata(
    data: DatasetBundle,
    experiment: ExperimentConfig,
    seed_info: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = environment_metadata()
    metadata.update(
        {
            "dataset": "mnist",
            "dataset_seed": experiment.dataset_seed,
            "split_id": data.split_id,
            "training_split_size": int(len(data.x_train)),
            "validation_split_size": int(len(data.x_validation)),
            "test_split_size": int(len(data.x_test)),
            "input_shape": list(data.x_train.shape[1:]),
            "run_seed": experiment.run_seed,
            "trial_seed": seed,
            "mode": experiment.mode,
            "mode_interpretation": mode_interpretation(experiment.mode),
            "budget": experiment.budget_name,
            "max_epochs": experiment.max_epochs,
            "early_stopping_monitor": "val_accuracy",
            "early_stopping_mode": "max",
            "early_stopping_patience": experiment.early_stopping_patience,
            "early_stopping_min_delta": 0.0,
            "restore_best_weights": True,
        }
    )
    if seed_info:
        metadata.update(seed_info)
    return metadata


def evaluate_candidate(
    configuration: dict[str, Any],
    experiment: ExperimentConfig,
    data: DatasetBundle,
) -> EvaluationResult:
    seed = trial_seed(experiment.run_seed, experiment.mode, configuration)
    started = time.perf_counter()
    warning = _semantic_warning(configuration, experiment.mode)
    seed_info: dict[str, Any] | None = None
    try:
        seed_info = seed_tensorflow(seed, experiment.deterministic)
        model, effective_learning_rate = build_model(configuration, experiment.mode)
        import tensorflow as tf

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=experiment.early_stopping_patience,
            min_delta=0.0,
            restore_best_weights=True,
        )
        history = model.fit(
            data.x_train,
            _labels_for_loss(data.y_train, configuration, experiment.mode),
            validation_data=(
                data.x_validation,
                _labels_for_loss(data.y_validation, configuration, experiment.mode),
            ),
            epochs=experiment.max_epochs,
            batch_size=_training_batch_size(configuration),
            callbacks=[callback],
            verbose=0,
        )
        history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
        best_epoch, best_accuracy, best_loss = _history_best(history_dict)
        train_accuracy = _training_accuracy(history_dict, best_epoch)
        evaluation_metadata = _evaluation_metadata(data, experiment, seed_info, seed)
        evaluation_metadata["actual_epochs_completed"] = len(
            history_dict.get("accuracy", history_dict.get("loss", []))
        )
        return EvaluationResult(
            status="success",
            fitness=best_accuracy,
            train_accuracy=train_accuracy,
            validation_accuracy=best_accuracy,
            validation_loss=best_loss,
            best_epoch=best_epoch,
            training_time_seconds=time.perf_counter() - started,
            effective_learning_rate=effective_learning_rate,
            semantic_warning=warning,
            metadata=evaluation_metadata,
        )
    except Exception as error:  # A failed candidate must not stop the colony.
        return EvaluationResult(
            status="failed",
            fitness=None,
            train_accuracy=None,
            validation_accuracy=None,
            validation_loss=None,
            best_epoch=None,
            training_time_seconds=time.perf_counter() - started,
            failure_reason=f"{type(error).__name__}: {error}",
            metadata={
                **_evaluation_metadata(data, experiment, seed_info, seed),
                "actual_epochs_completed": None,
            },
        )
    finally:
        try:
            import tensorflow as tf

            try:
                tf.keras.backend.clear_session(free_memory=True)
            except TypeError:
                tf.keras.backend.clear_session()
            gc.collect()
        except (ImportError, AttributeError, RuntimeError):
            gc.collect()


def train_final_model(
    configuration: dict[str, Any],
    experiment: ExperimentConfig,
    data: DatasetBundle,
    best_epoch: int,
    output_path: str,
) -> dict[str, Any]:
    """Train on all selected development images and evaluate once on test."""
    started = time.perf_counter()
    development_size = len(data.x_train) + len(data.x_validation)
    if development_size < 10:
        raise ValueError("Final retraining requires at least 10 development images")
    if len(data.x_test) != 10_000:
        raise ValueError("Final evaluation requires exactly 10,000 test images")
    seed = trial_seed(experiment.run_seed, experiment.mode, configuration)
    seed_tensorflow(seed, experiment.deterministic)
    model, effective_learning_rate = build_model(configuration, experiment.mode)
    x_development = np.concatenate([data.x_train, data.x_validation], axis=0)
    y_development = np.concatenate([data.y_train, data.y_validation], axis=0)
    model.fit(
        x_development,
        _labels_for_loss(y_development, configuration, experiment.mode),
        epochs=best_epoch,
        batch_size=_training_batch_size(configuration),
        shuffle=True,
        verbose=0,
    )
    loss, accuracy = model.evaluate(
        data.x_test,
        _labels_for_loss(data.y_test, configuration, experiment.mode),
        verbose=0,
    )
    model.save(output_path)
    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "best_epoch": best_epoch,
        "effective_learning_rate": effective_learning_rate,
        "model_path": output_path,
        "runtime_seconds": time.perf_counter() - started,
    }
