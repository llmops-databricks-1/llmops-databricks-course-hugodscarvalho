"""Shared Databricks notebook utilities.

These helpers paper over the differences between local execution (where
``dbutils`` is unavailable) and Databricks runtime execution, keeping
notebook and deployment-script code free of try/except boilerplate.
"""

from __future__ import annotations

import os

import mlflow
from dotenv import load_dotenv


def get_widget(name: str, default: str | None = None) -> str | None:
    """Return the value of a Databricks widget, falling back to ``default``.

    Safe to call both inside a Databricks runtime (where ``dbutils`` is
    injected as a built-in) and in local / CI environments (where it is not).

    Args:
        name: Widget name.
        default: Value to return when the widget is not set or when running
            outside a Databricks cluster.

    Returns:
        Widget value, or ``default`` if unavailable.
    """
    try:
        # ``dbutils`` is a Databricks built-in — not importable locally.
        from databricks.sdk.runtime import dbutils  # type: ignore[import]

        return dbutils.widgets.get(name)
    except Exception:
        return default


def set_mlflow_tracking_uri() -> None:
    """Configure MLflow to talk to the Databricks tracking server.

    Only needed for **local** execution.  Inside a Databricks cluster the
    tracking URI is already configured via the runtime environment.

    Reads ``PROFILE`` from the environment (or ``.env`` file) and sets both
    the tracking URI and the Unity Catalog registry URI.
    """
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        load_dotenv()
        profile = os.environ["PROFILE"]
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
