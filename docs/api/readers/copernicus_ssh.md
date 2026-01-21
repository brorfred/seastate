# seastate.readers.copernicus_ssh

File handler for Copernicus Marine Service SSH NetCDF files.

## Overview

This module provides a SatPy-compatible file handler for reading Copernicus
Marine Service Sea Surface Height (SSH) data from NetCDF files.

## Example Usage

```python
from satpy import Scene

# Load SSH data using the custom reader
scn = Scene(
    filenames=["copernicus_SSH_2026-01-15.nc"],
    reader='copernicus_ssh'
)
scn.load(['sla', 'adt', 'ugos', 'vgos'])

# Access data
sla = scn['sla']
print(sla)
```

## API Reference

::: seastate.readers.copernicus_ssh
    options:
      show_root_heading: false
      show_root_full_path: false
