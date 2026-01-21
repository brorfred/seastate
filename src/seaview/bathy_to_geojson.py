import matplotlib.pyplot as plt
import numpy as np
import json

# Create sample data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create contour plot
contours = plt.contour(X, Y, Z)

# Convert to GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": []
}

# Minimum number of points per contour line
min_points = 50

# Extract each contour line using allsegs
for level_idx, level in enumerate(contours.levels):
    segments = contours.allsegs[level_idx]

    for segment in segments:
        # Skip segments with fewer than min_points
        if len(segment) < min_points:
            continue

        # Create a GeoJSON LineString feature
        feature = {
            "type": "Feature",
            "properties": {
                "level": float(level),
                "level_index": level_idx,
                "num_points": len(segment)  # Optional: store the point count
            },
            "geometry": {
                "type": "LineString",
                "coordinates": segment.tolist()
            }
        }
        geojson["features"].append(feature)

# Save to file
with open('contours.geojson', 'w') as f:
    json.dump(geojson, f, indent=2)

print(f"Saved {len(geojson['features'])} contour lines to contours.geojson")
