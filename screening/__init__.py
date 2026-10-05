"""Hybrid screening package for the Sechenov anemia case."""

# Конфигурация должна быть загружена до импорта модулей, читающих os.getenv
# (casebase, rag, api), поэтому импортируем её первой.
from . import config  # noqa: F401

__version__ = "5.12.1"
