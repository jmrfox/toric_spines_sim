"""Pytest configuration and shared fixtures for toric_spines_sim tests."""

import tempfile
from pathlib import Path
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_swc_file(temp_dir):
    """Create a minimal valid SWC file for testing."""
    swc_content = """# Sample SWC file for testing
# CYCLE_BREAK reconnect 3 4
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
3 1 2.0 0.0 0.0 0.6 2
4 1 2.0 0.0 0.0 0.6 2
5 1 3.0 0.0 0.0 0.5 3
"""
    swc_path = temp_dir / "test.swc"
    swc_path.write_text(swc_content)
    return swc_path


@pytest.fixture
def sample_swc_with_sink(temp_dir):
    """Create a SWC file with sink annotation."""
    swc_content = """# Sample SWC file with sink
# SINK: start=1, end=3, axis=x, length=100.0, radius=0.5, tag=5, neck_xyz=0.000000 0.000000 0.000000
1 1 0.0 0.0 0.0 1.0 -1
2 1 1.0 0.0 0.0 0.8 1
3 1 2.0 0.0 0.0 0.6 2
"""
    swc_path = temp_dir / "test_sink.swc"
    swc_path.write_text(swc_content)
    return swc_path


@pytest.fixture
def sample_synpts_file(temp_dir):
    """Create a sample synapse points file."""
    synpts = np.array(
        [
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ]
    )
    synpts_path = temp_dir / "synpts.txt"
    np.savetxt(synpts_path, synpts)
    return synpts_path


@pytest.fixture
def sample_nff_file(temp_dir):
    """Create a sample NFF file with s-points."""
    nff_content = """# Sample NFF file
# object with 1 contours.
f 1 1 1 0 0 0 0 0
s 7886 953.333 3033.91 20
s 7831.33 1055.67 3061.74 20
s 7800.0 1100.0 3080.0 20
"""
    nff_path = temp_dir / "test.nff"
    nff_path.write_text(nff_content)
    return nff_path


@pytest.fixture
def sample_neck_points_file(temp_dir):
    """Create a sample neck points file."""
    neck_points = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.1, 0.1, 0.0],
        ]
    )
    neck_path = temp_dir / "neck_points.txt"
    np.savetxt(neck_path, neck_points)
    return neck_path
