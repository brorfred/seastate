# seastate.data_sources.globcolour

GlobColour chlorophyll data access.

## Overview

This module provides functions to retrieve and access GlobColour merged
chlorophyll-a concentration data from the Copernicus Marine Service.

**Product**: OCEANCOLOUR_GLO_BGC_L3_NRT_009_101

## Example Usage

```python
from seastate.data_sources import globcolour

# Open dataset for a specific date
ds = globcolour.open_dataset(dtm="2026-01-15")

# Access chlorophyll (mg/m³)
chl = ds.CHL
print(f"Chl range: {float(chl.min()):.3f} to {float(chl.max()):.1f} mg/m³")
```

## API Reference

::: seastate.data_sources.globcolour
    options:
      show_root_heading: false
      show_root_full_path: false
