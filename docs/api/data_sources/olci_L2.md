# seastate.data_sources.olci_L2

OLCI Level-2 ocean colour data access from EUMETSAT.

## Overview

This module provides functions to retrieve and process Sentinel-3 OLCI
Level-2 ocean colour products from the EUMETSAT Data Store.

**Product**: EO:EUM:DAT:0407

## Example Usage

```python
from seastate.data_sources import olci_L2

# Retrieve swaths for a date
olci_L2.retrieve(dtm="2026-01-15")

# List downloaded swaths
swaths = olci_L2.swathlist(dtm="2026-01-15")
print(f"Found {len(swaths)} swaths")

# Extract and load as Satpy scenes
scenes = olci_L2.extract_swath_scenes(dtm="2026-01-15")
```

!!! warning "Credentials Required"
    EUMETSAT data access requires credentials stored in `~/.eumdac/credentials`.

## API Reference

::: seastate.data_sources.olci_L2
    options:
      show_root_heading: false
      show_root_full_path: false
