from numpy import random
import pytest

from spline import STANDARDS, Spline


def standard() -> str:
    return random.choice(('1139', '6033', '100092'))


def join(standard) -> str:
    if standard == '1139':
        return random.choice(('inner', 'outer', 'left', 'right'))
    elif standard == '6033':
        return random.choice(('outer', 'left', 'right'))
    elif standard == '100092':
        return random.choice(('left', 'right'))

def parameters(standard) -> dict:
    row = STANDARDS[standard]['standard'].sample(n=1)
    if standard == '1139':
        return {'n_teeth': int(row['n_teeth'].iloc[0]), 'd': float(row['d'].iloc[0]), 'D': float(row['D'].iloc[0])}
    elif standard == '6033':
        row = row.loc[:, (row != 0).all(axis=0)]
        module = random.choice(row.columns)
        return {'n_teeth': int(row[module].iloc[0]), 'module': float(module), 'D': float(row.index[0])}
    elif standard == '100092':
        return {'n_teeth': int(row['n_teeth'].iloc[0]), 'module': float(row['module'].iloc[0]), 'd': float(row['d'].iloc[0])}

@pytest.fixture(scope='function')
def spline():
    s = standard()
    return Spline(s, join(s), **parameters(s))

def moment():
    return random.uniform(-800, 800)

def length():
    return random.uniform(3, 50) / 1_000

def max_tension():
    return random.uniform(3, 50) * 10 ** 6