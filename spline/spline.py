from types import MappingProxyType
import os
import sys

import numpy as np
from numpy import linspace, pi, sin, cos
import pandas as pd
from matplotlib import pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)

gost_1139_light_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Легкая серия')
print(gost_1139_light_series)
gost_1139_middle_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Средняя серия')
print(gost_1139_middle_series)
gost_1139_heavy_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Тяжелая серия')
print(gost_1139_heavy_series)

gost_1139 = pd.concat([gost_1139_light_series, gost_1139_middle_series, gost_1139_heavy_series], axis=0)

REFERENCES = MappingProxyType({
    1: '''Детали машин: учебник для вузов /
    [Л.А. Андриенко, Д38 Б.А. Байков, М.Н. Захаров и др.]; под ред. О.А. Ряховского. -
    4-е изд., перераб. и доп. - 
    Москва: Издательство МГТУ им. Н.Э. Баумана, 2014. - 465, [7] с.: ил''',
    2: '''''',

})


class Spline:
    """Шлицевое соединение"""

    TYPES = {1139: 'прямобочные шлицевые соединения',
             6033: 'шлицевые соединения с эвольвентными зубьями',
             100092: 'шлицевые соединения треугольного профиля'}

    # вид центрирования
    CENTERING = {-1: 'по внутреннему диаметеру',
                 0: 'по боковым граням',
                 +1: 'по наружному диаметру'}

    def __init__(self, gost: int, centering: int):
        assert gost in Spline.TYPES.keys()
        assert centering in Spline.CENTERING.keys()

        self.gost = gost
        self.centering = centering  # вид центрирования

    def __slit_1139(self):
        pass

    @property
    def average_diameter(self) -> float:
        """Средний диаметр"""
        return 0.5 * (self.d + self.D) - 2 * self.f
        return self.D - 1.1 * self.m
        return self.m * self.z

    def tension(self):
        return 2 * T * k / (d_ * z * h * l)

    def t(self, max_tension, moment, length, safety=1) -> tuple[dict[str: float], ...]:
        result = list()
        for d, D, z in zip():
            d_ = (d + D) / 2
            tension = 2 * moment * k / (d_ * z * h * length)
            if tension <= max_tension: result.append({'d': d, 'D': D, 'z': z})
        return tuple(result)

    def show(self, **kwargs):
        """Визуализация сечения шлица"""
        plt.figure(figsize=kwargs.pop('figsize', (4, 4)))
        angle = linspace(0, 2 * pi, 360, endpoint=True, dtype='float32')
        plt.plot([0, 0], [-self.D, self.D], color='orange', )
        plt.plot([-self.D, self.D], [0, 0], color='orange', )
        plt.plot(self.d / 2 * cos(angle), self.d / 2 * sin(angle), color='black', ls='solid', linewidth=3)
        plt.plot(self.D / 2 * cos(angle), self.D / 2 * sin(angle), color='black', ls='solid', linewidth=3)
        plt.tight_layout()
        plt.show()


def test():
    splines = list()

    if 1:
        splines.append(Spline(1139, -1))

    for spline in splines:
        pass


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
