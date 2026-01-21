# seastate.data_sources.gebco_bathy

GEBCO Bathymetry data access.

## Overview

This module provides functions to retrieve and access GEBCO bathymetry
data from CEDA via OPeNDAP.

**Product**: GEBCO 2025 Sub-Ice Topography

## Example Usage

```python
from seastate.data_sources import gebco_bathy

# Open dataset (bathymetry is time-independent)
ds = gebco_bathy.open_dataset()

# Access elevation data (negative values = ocean depth)
elevation = ds.elevation
print(f"Depth range: {float(elevation.min()):.0f} to {float(elevation.max()):.0f} m")

# Force re-download
gebco_bathy.retrieve(force=True)
```

## API Reference

::: seastate.data_sources.gebco_bathy
    options:
      show_root_heading: false
      show_root_full_path: false
