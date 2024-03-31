import pathlib as plib
import pandas as pd
import numpy as np
import pytest
from tga_data_analysis.tga import Project, Sample

test_dir: plib.Path = plib.Path(__file__).parent / "data"


@pytest.fixture
def project():
    # Create a mock Project instance
    return Project(test_dir, name="test")


@pytest.fixture
def sample_instance(project):
    # Create a Sample instance with mock data
    return Sample(project=project, name="test_sample", filenames=["dummy_filename"])


@pytest.fixture
def sample_file():
    # Create a sample DataFrame to simulate the file data
    return pd.DataFrame(
        {
            "T_C": [25, 50, 75],  # Temperatures in Celsius
            "m_mg": [200, 150, 100],  # Mass in mg
            "m_p": [100, 75, 50],  # Mass percentage
        }
    )
