"""imagenette-cnn package.

CNN model for imagenette data.
"""

from __future__ import annotations

from imagenette_cnn._internal.cli import get_parser, main

__all__: list[str] = ["get_parser", "main"]
