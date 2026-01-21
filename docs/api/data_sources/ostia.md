# seastate.data_sources.ostia

OSTIA Sea Surface Temperature data access.

## Overview

This module provides functions to retrieve and access OSTIA
(Operational Sea Surface Temperature and Ice Analysis) data
from the Copernicus Marine Service.

**Product**: SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001

## Example Usage

```python
from seastate.data_sources import ostia

# Open dataset for a specific date
ds = ostia.open_dataset(dtm="2026-01-15")

# Access SST (in Kelvin, convert to Celsius)
sst_celsius = ds.analysed_sst - 273.15
print(f"SST range: {float(sst_celsius.min()):.1f} to {float(sst_celsius.max()):.1f} °C")
```

## API Reference

::: seastate.data_sources.ostia
    options:
      show_root_heading: false
      show_root_full_path: false
