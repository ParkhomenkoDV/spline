from numpy import random
import pytest

from spline import Spline


@pytest.fixture(scope='function')
def standard() -> int:
    return random.choice((1139, 6033, 100092))


@pytest.fixture(scope='function')
def join() -> str:
    return random.choice(('inner', 'outer', 'left', 'right'))


@pytest.fixture(scope='function')
def spline(standard):
    if standard == 1139:
        return Spline(standard, random.choice(('inner', 'outer', 'left', 'right')))
    elif standard == 6033:
        return Spline(standard, random.choice(('outer', 'left', 'right')))
