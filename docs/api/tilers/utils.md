# seastate.tilers.utils

Utility functions for tile generation.

## Overview

This module provides helper functions for processing and filtering
contour data during tile generation.

## Example Usage

```python
from seastate.tilers.utils import filter_small_contours
import matplotlib.pyplot as plt
from matplotlib import tri

# After creating a tricontour plot
cs = ax.tricontour(triang, data, levels=10)

# Remove small contour segments (less than 100 vertices)
filter_small_contours(cs, min_vertices=100)
```

## API Reference

::: seastate.tilers.utils
    options:
      show_root_heading: false
      show_root_full_path: false
