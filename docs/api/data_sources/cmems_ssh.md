# seastate.data_sources.cmems_ssh

CMEMS Sea Surface Height (SSH) data access.

## Overview

This module provides functions to retrieve and access Sea Surface Height
data from the Copernicus Marine Service (CMEMS).

**Product**: SEALEVEL_GLO_PHY_L4_NRT_008_046

## Example Usage

```python
from seastate.data_sources import cmems_ssh

# Open dataset for a specific date
ds = cmems_ssh.open_dataset(dtm="2026-01-15")

# Access sea level anomaly
sla = ds.sla
print(f"SLA range: {float(sla.min()):.2f} to {float(sla.max()):.2f} m")

# Force re-download
cmems_ssh.retrieve(dtm="2026-01-15", force=True)
```

## API Reference

::: seastate.data_sources.cmems_ssh
    options:
      show_root_heading: false
      show_root_full_path: false
