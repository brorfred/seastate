"""Utility functions for tile generation.

This module provides helper functions for processing and filtering
contour data during tile generation.
"""
import numpy as np
import matplotlib
from matplotlib.path import Path


def filter_small_contours(cs, min_vertices=5):
    """Remove small contour segments from a TriContourSet.

    Filters out contour segments that have fewer vertices than the
    specified minimum, which helps clean up noisy or insignificant
    contour lines.

    Parameters
    ----------
    cs : matplotlib.tri._tricontour.TriContourSet
        The contour set to filter.
    min_vertices : int, optional
        Minimum number of vertices required for a segment to be kept,
        by default 5.
    """
    assert (type(cs) == matplotlib.tri._tricontour.TriContourSet)

    new_paths = []

    for level_idx, path in enumerate(cs.get_paths()):
        if len(path.vertices) == 0:
            new_paths.append(path)
            continue

        # Split path into individual segments using MOVETO codes
        segments = []
        current_segment = []

        for i, (vertex, code) in enumerate(zip(path.vertices, path.codes)):
            if code == Path.MOVETO:
                if current_segment:
                    segments.append(current_segment)
                current_segment = [vertex]
            else:
                current_segment.append(vertex)

        if current_segment:
            segments.append(current_segment)

        # Filter segments by vertex count
        filtered_segments = [seg for seg in segments if len(seg) >= min_vertices]

        # Rebuild path from filtered segments
        if filtered_segments:
            new_vertices = []
            new_codes = []
            for seg in filtered_segments:
                new_vertices.extend(seg)
                new_codes.append(Path.MOVETO)
                new_codes.extend([Path.LINETO] * (len(seg) - 1))

            new_paths.append(Path(new_vertices, new_codes))
        else:
            # Empty path needs proper 2D shape
            new_paths.append(Path(np.empty((0, 2)), []))

    cs.set_paths(new_paths)
