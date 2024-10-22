from numpy import random
import pytest

from spline import Spline


def test_init_spline(standard):
    if standard == 1139:
        assert Spline(standard, random.choice(('inner', 'outer', 'left', 'right')))
    elif standard == 6033:
        assert Spline(standard, random.choice(('outer', 'left', 'right')))
