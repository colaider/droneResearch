import genesis as gs
import pytest

@pytest.fixture(scope='session', autouse=True)
def genesis_cpu():
    gs.init(backend=gs.cpu, logging_level='error')
