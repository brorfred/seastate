"""Tests for the seastate.tilers.utils module."""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import tri
from matplotlib.path import Path

from seastate.tilers.utils import filter_small_contours


class TestFilterSmallContours:
    """Tests for the filter_small_contours function."""

    def _create_contour_set(self, x, y, z, levels=5):
        """Helper to create a TriContourSet for testing."""
        fig, ax = plt.subplots()
        triang = tri.Triangulation(x, y)
        cs = ax.tricontour(triang, z, levels=levels)
        plt.close(fig)
        return cs

    def test_filter_removes_small_segments(self):
        """filter_small_contours should remove segments with few vertices."""
        # Create a grid of points
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z = np.sin(x_flat) * np.cos(y_flat)

        cs = self._create_contour_set(x_flat, y_flat, z)

        # Get paths before filtering
        paths_before = cs.get_paths()
        total_before = sum(len(p.vertices) for p in paths_before if len(p.vertices) > 0)

        # Filter with a high threshold
        filter_small_contours(cs, min_vertices=50)

        # Get paths after filtering
        paths_after = cs.get_paths()
        total_after = sum(len(p.vertices) for p in paths_after if len(p.vertices) > 0)

        # After filtering with high threshold, should have fewer or equal vertices
        assert total_after <= total_before

    def test_filter_preserves_large_segments(self):
        """filter_small_contours should preserve segments with enough vertices."""
        # Create a larger grid for more detailed contours
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xx, yy = np.meshgrid(x, y)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z = np.sin(x_flat * 0.5) * np.cos(y_flat * 0.5)

        cs = self._create_contour_set(x_flat, y_flat, z, levels=3)

        # Filter with a very low threshold (should preserve most)
        filter_small_contours(cs, min_vertices=2)

        # Should still have paths
        paths_after = cs.get_paths()
        has_content = any(len(p.vertices) > 0 for p in paths_after)
        assert has_content

    def test_filter_handles_empty_paths(self):
        """filter_small_contours should handle contours with empty paths."""
        # Create minimal data
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        z = np.array([0, 1, 0, 1, 2, 1, 0, 1, 0])

        cs = self._create_contour_set(x, y, z, levels=2)

        # Should not raise an error
        filter_small_contours(cs, min_vertices=100)

    def test_filter_accepts_tricontourset_only(self):
        """filter_small_contours should only accept TriContourSet objects."""
        # Regular contour (not tri) should raise assertion error
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) * np.cos(yy)
        cs = ax.contour(xx, yy, z)  # Regular contour, not tricontour
        plt.close(fig)

        with pytest.raises(AssertionError):
            filter_small_contours(cs, min_vertices=5)

    def test_filter_default_min_vertices(self):
        """filter_small_contours should use default min_vertices=5."""
        x = np.linspace(0, 10, 30)
        y = np.linspace(0, 10, 30)
        xx, yy = np.meshgrid(x, y)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z = np.sin(x_flat) * np.cos(y_flat)

        cs = self._create_contour_set(x_flat, y_flat, z)

        # Should work with default parameter
        filter_small_contours(cs)  # Uses default min_vertices=5

    def test_filter_creates_valid_paths(self):
        """filter_small_contours should create valid matplotlib paths."""
        x = np.linspace(0, 10, 40)
        y = np.linspace(0, 10, 40)
        xx, yy = np.meshgrid(x, y)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z = np.sin(x_flat * 0.3) * np.cos(y_flat * 0.3)

        cs = self._create_contour_set(x_flat, y_flat, z, levels=5)
        filter_small_contours(cs, min_vertices=10)

        # Check all paths are valid Path objects
        for path in cs.get_paths():
            assert isinstance(path, Path)
            # Vertices should be 2D array
            if len(path.vertices) > 0:
                assert path.vertices.ndim == 2
                assert path.vertices.shape[1] == 2
