"""Path fixtures for oracle comparisons.

Two oracle sources:
  - Fortran: raw outputs in ``pwmva_package`` (authoritative for bit-compare)
  - Python:  ``pwmva_python`` numpy+numba implementation, already aligned to
             Fortran; cheaper to compare since we can run it in-process.
"""
from pathlib import Path
import sys
import pytest

PKG_ROOT = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_package")
PY_ROOT  = Path("/home/pisquare/zhijun/pwmva_fortran/pwmva_python")
VIZ_DIR  = Path(__file__).parent / "_viz_out"
VIZ_DIR.mkdir(exist_ok=True)

# Make the numpy reference ``pwmva`` package importable for oracle comparisons.
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))


@pytest.fixture(scope="session")
def pkg_root() -> Path:
    return PKG_ROOT


@pytest.fixture(scope="session")
def model_dir(pkg_root) -> Path:
    return pkg_root / "model" / "2D_models" / "mlayer"


@pytest.fixture(scope="session")
def csg_dir(pkg_root) -> Path:
    return pkg_root / "results" / "2D_models" / "mlayer" / "fdcsg_f40"


@pytest.fixture(scope="session")
def rerun_dir(pkg_root) -> Path:
    return pkg_root / "results" / "2D_models" / "mlayer" / "pwmva_warp_rerun"


@pytest.fixture(scope="session")
def oracle_dir(pkg_root) -> Path:
    return pkg_root / "results" / "2D_models" / "mlayer" / "cpg_oracle"


@pytest.fixture(scope="session")
def working_dir(pkg_root) -> Path:
    return pkg_root / "working"


@pytest.fixture(scope="session")
def viz_dir() -> Path:
    return VIZ_DIR
