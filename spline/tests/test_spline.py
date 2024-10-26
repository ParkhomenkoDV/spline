from numpy import random
import pytest

from conftest import standard, join, parameters, moment, length, max_tension

from spline import STANDARDS, Spline


def test_init_spline1139():
    s = '1139'
    for j in ('left', 'right', 'inner', 'outer'):
        for _, row in STANDARDS[s]['standard'].iterrows():
            assert Spline(s, j, n_teeth=row['n_teeth'], d=row['d'], D=row['D'])

def test_init_spline6033():
    s = '6033'
    for j in ('left', 'right', 'outer'):
        for D, row in STANDARDS[s]['standard'].iterrows():
            row=row[row > 0].to_dict()
            for module, n_teeth in row.items():
                assert Spline(s, j, n_teeth=n_teeth, module=module, D=D)

def test_init_spline100092():
    s = '100092'
    for j in ('left', 'right'):
        for _, row in STANDARDS[s]['standard'].iterrows():
            assert Spline(s, j, n_teeth=row['n_teeth'], module=row['module'], d=row['d'])

def test_init_spline():
    for _ in range(1_000):
        s = standard()
        assert Spline(s, join(s), **parameters(s))


def test_spline_tension():
    for _ in range(1_000):
        s = standard()
        sp = Spline(s, join(s), **parameters(s))
        assert sp.tension(moment(), length())


def test_fit_spline():
    for s in STANDARDS.keys():
        if s == '1139':
            joins = ('left', 'right', 'inner', 'outer')
        elif s == '6033':
            joins = ('left', 'right', 'outer')
        elif s == '100092': 
            joins = ('left', 'right')
        for j in joins:
            assert len(Spline.fit(s, j, max_tension(), moment(), length())) >= 0


