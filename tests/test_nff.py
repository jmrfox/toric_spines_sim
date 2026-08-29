"""Tests for toric_spines_sim.nff module."""

import pytest
import numpy as np
from pathlib import Path
from toric_spines_sim.utils import read_nff_s_points, read_nff_points_and_write_txt_file


class TestReadNffSPoints:
    """Test suite for read_nff_s_points function."""

    def test_read_basic(self, sample_nff_file):
        """Test reading basic NFF file."""
        points = read_nff_s_points(sample_nff_file)
        assert len(points) == 3
        assert isinstance(points, list)
        assert all(len(p) == 3 for p in points)

    def test_read_as_numpy(self, sample_nff_file):
        """Test reading NFF file as numpy array."""
        points = read_nff_s_points(sample_nff_file, return_numpy=True)
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)
        assert points.dtype == float

    def test_read_empty_file(self, temp_dir):
        """Test reading empty NFF file."""
        empty_file = temp_dir / "empty.nff"
        empty_file.write_text("")
        points = read_nff_s_points(empty_file)
        assert points == []

    def test_read_no_s_lines(self, temp_dir):
        """Test reading NFF file with no s-lines."""
        nff_content = """# Comment line
f 1 1 1 0 0 0 0 0
v 1.0 2.0 3.0
"""
        nff_file = temp_dir / "no_s.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert points == []

    def test_read_with_comments(self, temp_dir):
        """Test reading NFF file with comments."""
        nff_content = """# This is a comment
s 1.0 2.0 3.0 10
# Another comment
s 4.0 5.0 6.0 20
"""
        nff_file = temp_dir / "with_comments.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert len(points) == 2
        assert points[0] == (1.0, 2.0, 3.0)
        assert points[1] == (4.0, 5.0, 6.0)

    def test_read_case_insensitive(self, temp_dir):
        """Test that s-line parsing is case insensitive."""
        nff_content = """S 1.0 2.0 3.0 10
s 4.0 5.0 6.0 20
"""
        nff_file = temp_dir / "case_test.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert len(points) == 2

    def test_read_malformed_line_skipped(self, temp_dir):
        """Test that malformed lines are skipped."""
        nff_content = """s 1.0 2.0 3.0 10
s invalid data here
s 4.0 5.0 6.0 20
s 7.0 8.0
"""
        nff_file = temp_dir / "malformed.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert len(points) == 2
        assert points[0] == (1.0, 2.0, 3.0)
        assert points[1] == (4.0, 5.0, 6.0)

    def test_read_with_extra_fields(self, temp_dir):
        """Test reading s-lines with extra fields (index is ignored)."""
        nff_content = """s 1.0 2.0 3.0 10 extra field
s 4.0 5.0 6.0 20
"""
        nff_file = temp_dir / "extra_fields.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert len(points) == 2

    def test_read_string_path(self, sample_nff_file):
        """Test reading with string path instead of Path object."""
        points = read_nff_s_points(str(sample_nff_file))
        assert len(points) == 3

    def test_read_preserves_order(self, temp_dir):
        """Test that point order is preserved."""
        nff_content = """s 1.0 0.0 0.0 10
s 2.0 0.0 0.0 10
s 3.0 0.0 0.0 10
"""
        nff_file = temp_dir / "order_test.nff"
        nff_file.write_text(nff_content)
        points = read_nff_s_points(nff_file)
        assert points[0][0] == 1.0
        assert points[1][0] == 2.0
        assert points[2][0] == 3.0


class TestReadNffPointsAndWriteTxtFile:
    """Test suite for read_nff_points_and_write_txt_file function."""

    def test_write_txt_file(self, sample_nff_file, temp_dir):
        """Test writing NFF points to txt file."""
        output_file = temp_dir / "output.txt"
        read_nff_points_and_write_txt_file(sample_nff_file, output_file)
        assert output_file.exists()
        data = np.loadtxt(output_file)
        assert data.shape[0] == 3
        assert data.shape[1] == 3

    def test_write_preserves_data(self, sample_nff_file, temp_dir):
        """Test that written data matches read data."""
        output_file = temp_dir / "output.txt"
        original_points = read_nff_s_points(sample_nff_file, return_numpy=True)
        read_nff_points_and_write_txt_file(sample_nff_file, output_file)
        written_points = np.loadtxt(output_file)
        np.testing.assert_array_almost_equal(original_points, written_points, decimal=3)
