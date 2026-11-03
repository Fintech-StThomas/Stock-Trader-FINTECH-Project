"""Nikhil's announcement collection interface."""
from .config import SourceConfig, load_sources
from .pipeline import collect

__all__ = ['SourceConfig', 'load_sources', 'collect']
