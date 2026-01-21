"""SatPy readers for various oceanographic data formats.

This package contains custom SatPy file handlers for reading data from
various oceanographic and satellite data sources.
"""

from processor.readers.copernicus_ssh import CopernicusSSHFileHandler

__all__ = ["CopernicusSSHFileHandler"]
