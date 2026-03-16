"""Basic tests to ensure the package is properly installed."""

import importlib

from loguru import logger


def test_package_import() -> None:
    """Test that the package can be imported."""
    package_name = "eu_policy_agent"
    logger.info(f"Attempting to import package: {package_name}")
    module = importlib.import_module(package_name)
    assert module is not None
    logger.info(f"Package '{package_name}' imported successfully: {module}")


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    package_name = "eu_policy_agent"
    logger.info(f"Checking __version__ attribute on package: {package_name}")
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
    logger.info(f"Package version: {module.__version__}")
