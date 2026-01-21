# processor.tilers.ssh_tiles_fast

Fast SSH tile generation utilities.

## Overview

This module provides optimized functions for generating SSH slippy map
tiles with per-tile resampling for accurate alignment. Each tile is
individually resampled to ensure perfect coordinate alignment with
the slippy map grid.

## Example Usage

```python
from processor.tilers.ssh_tiles_fast import satpy_ssh_to_tiles_fixed
from processor.data_sources import cmems_ssh

# Load scene
scn = cmems_ssh.open_scene(dtm="2026-01-15")

# Generate tiles with accurate per-tile alignment
satpy_ssh_to_tiles_fixed(
    scn,
    dtm="2026-01-15",
    min_zoom=0,
    max_zoom=8,
    workers=8,
    add_contours=True,
    log_qc_path="tile_qc.csv"
)
```

## API Reference

::: processor.tilers.ssh_tiles_fast
    options:
      show_root_heading: false
      show_root_full_path: false
