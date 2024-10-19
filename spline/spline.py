from types import MappingProxyType
import os
import sys

import numpy as np
from numpy import linspace, pi, sin, cos
import pandas as pd
from matplotlib import pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)

gost_1139_light_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Легкая серия', )
gost_1139_middle_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Средняя серия', )
gost_1139_heavy_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Тяжелая серия', )
gost_1139 = pd.concat([gost_1139_light_series, gost_1139_middle_series, gost_1139_heavy_series], axis=0)
gost_1139 = gost_1139.rename(columns={'z': 'n_teeth',
                                      'b': 'width',
                                      'd1': 'corner_diameter', 'a': 'corner_width',
                                      'c': 'chamfer', 'dc': 'delta_chamfer', 'r': 'radius'})
for column in ('d', 'D', 'width', 'corner_diameter', 'corner_width', 'chamfer', 'delta_chamfer', 'radius'):
    gost_1139[column] /= 1_000  # СИ

gost_6033 = pd.read_excel(os.path.join(HERE, '6033.xlsx'))  # TODO
print(gost_6033)

REFERENCES = MappingProxyType({
    1: '''Детали машин: учебник для вузов /
    [Л.А. Андриенко, Д38 Б.А. Байков, М.Н. Захаров и др.]; под ред. О.А. Ряховского. -
    4-е изд., перераб. и доп. - 
    Москва: Издательство МГТУ им. Н.Э. Баумана, 2014. - 465, [7] с.: ил''',
    2: '''''',

})


class Spline:
    """Шлицевое соединение"""
    __slots__ = ('__standard',
                 '__n_teeth', '__d', '__D',
                 '__chamfer', '__width', '__corner_diameter', '__corner_width', '__delta_chamfer', '__radius',
                 '__module',)

    TYPES = {1139: 'прямобочные шлицевые соединения',
             6033: 'шлицевые соединения с эвольвентными зубьями',
             100092: 'шлицевые соединения треугольного профиля'}

    # вид центрирования
    CENTERING = {-1: 'по внутреннему диаметру',
                 0: 'по боковым граням',
                 +1: 'по наружному диаметру'}

    def __init__(self, standard: int | np.integer, centering: int, **parameters):
        assert standard in Spline.TYPES.keys()
        assert centering in Spline.CENTERING.keys()

        self.__standard: int = int(standard)
        # self.centering = centering  # вид центрирования

        if standard == 1139:
            for parameter in ('n_teeth', 'd', 'D'):
                assert parameters.get(parameter, None) is not None, f'parameter {parameter} is None'
            n_teeth, d, D = parameters['n_teeth'], parameters['d'], parameters['D']
            for _, row in gost_1139.iterrows():
                if n_teeth == row['n_teeth'] and d == row['d'] and D == row['D']:
                    for key, value in row.to_dict().items():
                        setattr(self, f'_{self.__class__.__name__}__{key}', value)
                    break
            else:
                raise Exception(
                    f'n_teeth={parameters["n_teeth"]}, d={parameters["d"]}, D={parameters["D"]} not in GOST {standard}')
        elif standard == 6033:
            pass

    @property
    def standard(self) -> int:
        return self.__standard

    @property
    def n_teeth(self) -> int:
        return self.__n_teeth

    @property
    def d(self) -> float:
        return self.__d

    @property
    def D(self) -> float:
        return self.__D

    @property
    def module(self) -> float:
        return self.__module

    @property
    def chamfer(self) -> float:
        return self.__chamfer

    @property
    def average_diameter(self) -> float:
        """Средний диаметр [1, с.127]"""
        if self.standard == 1139: return 0.5 * (self.d + self.D) - 2 * self.chamfer
        if self.standard == 6033: return self.D - 1.1 * self.module
        if self.standard == 100092: return self.module * self.n_teeth

    @property
    def height(self) -> float:
        """Высота контакта [1, с.127]"""
        if self.standard == 1139: return (self.D - self.d) / 2 - 2 * self.chamfer
        if self.standard == 6033: return 0.8 * self.module
        if self.standard == 100092: return (self.D - self.d) / 2

    def tension(self, moment, length, unevenness) -> float:
        """Напряжение смятия [1, с.126]"""
        assert isinstance(moment, (int, float, np.number))
        assert isinstance(length, (int, float, np.number)) and 0 < length
        assert isinstance(unevenness, (int, float, np.number))
        return 2 * moment * unevenness / (self.average_diameter * self.n_teeth * self.height * length)

    @classmethod
    def fit(cls, standard: int | np.integer,
            max_tension: int | float | np.number,
            moment: int | float | np.number, length: int | float | np.number,
            safety: int | float | np.number = 1, unevenness=1.5) -> tuple[dict[str: float], ...]:
        """Подбор шлицевого соединения [1, с.126]"""
        result = list()
        for i, row in gost_1139.iterrows():
            spline = Spline(standard, -1, n_teeth=row['n_teeth'], d=row['d'], D=row['D'])
            tension = spline.tension(moment, length, unevenness)
            if tension * safety <= max_tension:
                result.append({'d': row['d'], 'D': row['D'], 'n_teeth': row['n_teeth'], 'safety': 0})
        return tuple(result)

    def show(self, **kwargs):
        """Визуализация сечения шлица"""
        plt.figure(figsize=kwargs.pop('figsize', (4, 4)))
        plt.plot([0, 0], [-self.D, self.D], color='orange', )
        plt.plot([-self.D, self.D], [0, 0], color='orange', )
        angle = linspace(0, 2 * pi, 360, endpoint=True, dtype='float32')
        plt.plot(self.d / 2 * cos(angle), self.d / 2 * sin(angle), color='black', ls='solid', linewidth=3)
        plt.plot(self.D / 2 * cos(angle), self.D / 2 * sin(angle), color='black', ls='solid', linewidth=3)
        plt.tight_layout()
        plt.show()


def test():
    splines = list()

    if 1:
        splines.append(Spline(1139, -1, n_teeth=6, d=23 / 1000, D=26 / 1000))
        print(splines[-1].tension(50, 0.03, 1.5))

    for spline in splines:
        fitted_splines = Spline.fit(spline.standard, 40 * 10 ** 6, 20, 20 / 1000, 1)
        for s in fitted_splines: print(s)

        spline = Spline(spline.standard, -1, **fitted_splines[0])


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
