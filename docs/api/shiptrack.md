# shiptrack

Ship track visualization module.

## Overview

This module provides functionality to visualize ship cruise tracks on
interactive Folium maps from CSV position data.

## Example Usage

```python
from shiptrack import ShipTrackMapper

# Create mapper from CSV with position data
mapper = ShipTrackMapper(
    "cruise_positions.csv",
    lat_col="latitude",
    lon_col="longitude"
)

# Get track center
center = mapper.get_center()
print(f"Track center: {center}")

# Create interactive map
m = mapper.create_map(
    zoom_start=8,
    line_color="darkblue",
    line_weight=3,
    show_markers=True,
    marker_interval=10
)

# Save to HTML
mapper.save_map("cruise_track.html")
```

## CSV Format

The input CSV should contain latitude and longitude columns:

```csv
lat,lon,timestamp
37.7749,-122.4194,2024-01-01T00:00:00
37.8044,-122.2712,2024-01-01T01:00:00
37.8716,-122.2727,2024-01-01T02:00:00
```

## API Reference

::: shiptrack
    options:
      show_root_heading: false
      show_root_full_path: false
