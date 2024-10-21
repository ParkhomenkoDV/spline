from types import MappingProxyType
import os
import sys

import numpy as np
from numpy import array, linspace, sqrt, nan, isnan, pi, sin, cos, arcsin as asin
import pandas as pd
from matplotlib import pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)

# dtype разрушает точность
gost_1139_light_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Легкая серия', )
gost_1139_middle_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Средняя серия', )
gost_1139_heavy_series = pd.read_excel(os.path.join(HERE, '1139.xlsx'), sheet_name='Тяжелая серия', )
gost_1139 = pd.concat([gost_1139_light_series, gost_1139_middle_series, gost_1139_heavy_series], axis=0)
gost_1139 = gost_1139.rename(columns={'z': 'n_teeth',
                                      'b': 'width',
                                      'd1': 'corner_diameter', 'a': 'corner_width',
                                      'c': 'chamfer', 'dc': 'deviation_chamfer', 'r': 'radius'})
for column in ('d', 'D', 'width', 'corner_diameter', 'corner_width', 'chamfer', 'deviation_chamfer', 'radius'):
    gost_1139[column] /= 1_000  # СИ для расчетов

gost_6033 = pd.read_excel(os.path.join(HERE, '6033.xlsx'), sheet_name='common', )

gost_6033[0] = gost_6033[0] / 1_000  # перевод D в СИ
gost_6033 = gost_6033.set_index(0)
gost_6033.index.name = 'D'
gost_6033 = gost_6033.rename(columns={column: float(str(column).strip().replace(',', '.')) / 1_000
                                      for column in gost_6033.columns})
gost_6033 = gost_6033.fillna(0)
for column in gost_6033.columns: gost_6033[column] = gost_6033[column].astype('int32')

REFERENCES = MappingProxyType({
    1: '''Детали машин: учебник для вузов /
    [Л.А. Андриенко, Д38 Б.А. Байков, М.Н. Захаров и др.]; под ред. О.А. Ряховского. -
    4-е изд., перераб. и доп. - 
    Москва: Издательство МГТУ им. Н.Э. Баумана, 2014. - 465, [7] с.: ил''',
    2: '''''',

})


def rotate(point, angle):
    """Поворот точки"""
    x, y = point
    return x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)


def validate_join(join) -> str:
    """Валидация центрирования"""
    assert isinstance(join, str)
    join = join.strip().lower()
    assert join in ('inner', 'outer', 'left', 'right')  # центрирование внутреннее, внешнее, боковое
    return join


class Spline1139:
    """Шлицевое соединение по ГОСТ 1139"""
    __STANDARD = 1139

    def __init__(self, join: str, n_teeth: int, d: float, D: float, **kwargs):
        self._join = validate_join(join)

        for _, row in gost_1139.iterrows():
            if n_teeth == row['n_teeth'] and d == row['d'] and D == row['D']:
                for key, value in row.to_dict().items(): setattr(self, f'_{key}', value)
                break
        else:
            raise Exception(f'n_teeth={n_teeth}, d={d}, D={D} not in standard {Spline1139.__STANDARD}. '
                            f'Look spline.gost_{Spline1139.__STANDARD}')

    def __str__(self) -> str:
        """Условное обозначение"""
        mm = 1_000  # перевод в мм
        if self.join == 'inner':
            return f'd-{self.n_teeth:.0f}x{self.d * mm:.0f}x{self.D * mm:.0f}x{self.width * mm:.0f}'
        elif self.join == 'outer':
            return f'D-{self.n_teeth:.0f}x{self.d * mm:.0f}x{self.D * mm:.0f}x{self.width * mm:.0f}'
        else:
            return f'b-{self.n_teeth:.0f}x{self.d * mm:.0f}x{self.D * mm:.0f}x{self.width * mm:.0f}'

    @property
    def join(self) -> str:
        """Центрирование"""
        return self._join

    @property
    def n_teeth(self) -> int:
        """Количество зубьев"""
        return int(self._n_teeth)

    @property
    def d(self) -> float:
        """Внутренний диаметр"""
        return self._d

    @property
    def D(self) -> float:
        """Внешний диаметр"""
        return self._D

    @property
    def width(self) -> float:
        """Ширина зуба"""
        return self._width

    @property
    def corner_diameter(self) -> float:
        """Диаметр канавки"""
        return self._corner_diameter

    @property
    def corner_width(self) -> float:
        """Ширина канавки"""
        return self._corner_width

    @property
    def chamfer(self) -> float:
        """Фаска"""
        return self._chamfer

    @property
    def deviation_chamfer(self) -> tuple[float, float]:
        """Отклонение фаски"""
        return 0, self._deviation_chamfer

    @property
    def radius(self) -> float:
        """Радис скругления"""
        return self._radius

    @property
    def height(self) -> float:
        """Высота контакта [1, с.127]"""
        return (self.D - self.d) / 2 - 2 * self.chamfer

    @property
    def average_diameter(self) -> float:
        """Средний диаметр [1, с.127]"""
        return 0.5 * (self.d + self.D) - 2 * self.chamfer

    def show(self, **kwargs) -> None:
        """Визуализация сечения шлица"""
        mm = 1_000  # перевод в мм
        d, D, width, chamfer, radius = self.d * mm, self.D * mm, self.width * mm, self.chamfer * mm, self.radius * mm
        corner_diameter, corner_width = self.corner_diameter * mm, self.corner_width * mm
        r, R, cr = d / 2, D / 2, corner_diameter / 2

        arc_D = linspace(asin(-(width / 2 - chamfer) / R), asin((width / 2 - chamfer) / R), 360 // self.n_teeth,
                         dtype='float16')
        arc_cd = linspace(0, 2 * pi / self.n_teeth - asin((width + radius) / cr), 360 // self.n_teeth,
                          dtype='float16')
        arc_radius = linspace(pi, 3 * pi / 2, 90, endpoint=True, dtype='float32')
        '''if not isnan(corner_width):
            arc_corner = linspace(asin(-(corner_width / 2 / r)), asin(corner_width / 2 / r), 360 // self.n_teeth,
                                  dtype='float16')'''

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
        plt.suptitle(kwargs.pop('suptitle', 'Spline'), fontsize=16, fontweight='bold')
        plt.title(kwargs.pop('title', str(self)), fontsize=14)
        for angle in linspace(pi / 2, 5 * pi / 2, self.n_teeth + 1, endpoint=True):
            # оси
            plt.plot(*rotate(array(((0, 0), (0, R))), angle),
                     color='orange', ls='dashdot', linewidth=1.5)
            plt.plot(*rotate(array(((0, 0), (0, R))), angle + pi / self.n_teeth),
                     color='orange', ls='dashdot', linewidth=1.5)
            # впадины
            plt.plot(cr * cos(arc_cd + angle + asin((width / 2 + radius) / cr)),
                     cr * sin(arc_cd + angle + asin((width / 2 + radius) / cr)),
                     color='black', ls='solid', linewidth=2)
            # вершины зубья
            plt.plot(R * cos(arc_D + angle), R * sin(arc_D + angle),
                     color='black', ls='solid', linewidth=2)
            '''if not isnan(corner_width):
                plt.plot(*rotate(array((r * cos(arc_corner), r * sin(arc_corner))),
                                 angle=angle + pi / self.n_teeth),
                         color='black', ls='solid', linewidth=2)'''
            for lr in (-1, +1):
                # фаски
                plt.plot(*rotate(
                    array(((lr * (width / 2 - chamfer), lr * (width / 2)),
                           (R * cos((width / 2 - chamfer) / R), R * cos((width / 2 - chamfer) / R) - chamfer))),
                    angle - pi / 2),
                         color='black', ls='solid', linewidth=2)
                # боковые грани зубьев
                plt.plot(*rotate(
                    array(((lr * width / 2, lr * width / 2),
                           (R * cos((width / 2 - chamfer) / R) - chamfer, cr - radius))),
                    angle - pi / 2),
                         color='black', ls='solid', linewidth=2)
                # радиус скругления
                plt.plot(*rotate(array((lr * (radius * cos(arc_radius) + (width / 2 + radius)),
                                        radius * sin(arc_radius) + (cr - radius))),
                                 angle - pi / 2),
                         color='black', ls='solid', linewidth=2)
        plt.grid(kwargs.pop('grid', True))
        plt.axis('square')
        plt.xlabel(kwargs.pop('xlabel', 'mm'), fontsize=12), plt.ylabel(kwargs.pop('ylabel', 'mm'), fontsize=12)
        plt.tight_layout()
        plt.show()


class Spline6033:
    """Шлицевое соединение по ГОСТ 6033"""
    __STANDARD = 6033

    def __init__(self, join: str, n_teeth: int, module: float, D: float, **kwargs):
        self._join = validate_join(join)

        assert module in gost_6033.columns, f'module {module} not in standard {Spline6033.__STANDARD}'
        series = gost_6033[module]  # фильтрация по модулю
        dct = series[series > 0].to_dict()  # фильтрация по существованию и преобразование в словарь
        assert dct.get(D, nan) == n_teeth and not isnan(n_teeth), \
            (f'n_teeth={n_teeth}, module={module}, D={D} not in standard {Spline6033.__STANDARD}. '
             f'Look spline.gost_{Spline6033.__STANDARD}')

        self._n_teeth, self._module, self._D = n_teeth, module, D

    def __str__(self) -> str:
        """Условное обозначение"""
        mm = 1_000  # перевод в мм
        return f'{self.D * mm:.0f}x{self.module * mm:.0f} ГОСТ {Spline6033.__STANDARD}-80'

    @property
    def n_teeth(self) -> int:
        return self._n_teeth

    @property
    def module(self) -> float:
        return self._module

    @property
    def D(self) -> float:
        return self._D

    @property
    def d(self) -> float:
        """Диаметр делительной окружности"""
        return self.module * self.n_teeth

    @property
    def alpha(self) -> float:
        """Угол профиля зуба [рад]"""
        return pi / 6

    @property
    def circumferential_step(self) -> float:
        """Делительный окружной шаг зубьев"""
        return pi * self.module

    @property
    def height(self) -> float:
        """Высота контакта [1, с.127]"""
        return 0.8 * self.module

    @property
    def average_diameter(self) -> float:
        """Средний диаметр [1, с.127]"""
        return self.D - 1.1 * self.module


class Spline100092:
    """Шлицевое соединение по ОСТ 100092"""
    __STANDARD = 100092

    def __init__(self, **kwargs):
        pass

    @property
    def height(self) -> float:
        """Высота контакта [1, с.127]"""
        return (self.D - self.d) / 2

    @property
    def average_diameter(self) -> float:
        """Средний диаметр [1, с.127]"""
        return self.module * self.n_teeth


class Spline(Spline1139, Spline6033, Spline100092):
    """Шлицевое соединение"""
    # __slots__ = ('__standard', '__join')

    TYPES = {1139: 'прямобочные шлицевые соединения',
             6033: 'шлицевые соединения с эвольвентными зубьями',
             100092: 'шлицевые соединения треугольного профиля'}

    # вид центрирования
    JOIN = {'inner': 'по внутреннему диаметру',
            'left': 'по боковым граням',
            'right': 'по боковым граням',
            'outer': 'по наружному диаметру'}

    def __init__(self, standard: int | np.integer, join: str, **parameters):
        assert standard in Spline.TYPES.keys()
        self.__standard: int = int(standard)  # избавление от np.integer
        assert join in Spline.JOIN.keys(), f'join {join} not in {Spline.JOIN}'
        self.__join: str = join  # вид центрирования

        # Определение родительского класса
        if self.standard == 1139:
            Spline1139.__init__(self, join, **parameters)
        elif self.standard == 6033:
            Spline6033.__init__(self, join, **parameters)
        elif self.standard == 100092:
            Spline100092.__init__(self, **parameters)
        else:
            raise Exception(f'standard {standard} not in {Spline.TYPES}')

    @property
    def standard(self) -> int:
        return self.__standard

    @property
    def join(self) -> str:
        """Центрирования"""
        return self.__join

    def tension(self, moment, length, unevenness) -> float:
        """Напряжение смятия [1, с.126]"""
        assert isinstance(moment, (int, float, np.number))
        assert isinstance(length, (int, float, np.number)) and 0 < length
        assert isinstance(unevenness, (int, float, np.number))
        return 2 * moment * unevenness / (self.average_diameter * self.n_teeth * self.height * length)

    @classmethod
    def fit(cls, standard: int | np.integer, join: str,
            max_tension: int | float | np.number,
            moment: int | float | np.number, length: int | float | np.number,
            safety: int | float | np.number = 1, unevenness=1.5) -> tuple[dict[str: float], ...]:
        """Подбор шлицевого соединения [1, с.126]"""
        assert standard in Spline.TYPES.keys()
        result = list()
        if standard == 1139:
            for _, row in gost_1139.iterrows():
                spline = Spline(standard, join, n_teeth=row['n_teeth'], d=row['d'], D=row['D'])
                tension = spline.tension(moment, length, unevenness)
                if tension * safety <= max_tension:
                    result.append({'n_teeth': int(row['n_teeth']), 'd': float(row['d']), 'D': float(row['D']),
                                   'safety': max_tension / (tension * safety)})
        elif standard == 6033:
            for _, row in gost_6033.iterrows():
                spline = Spline(standard, join, )
        elif standard == 100092:
            pass
        return tuple(result)


def test():
    splines, conditions = list(), list()

    if 1139 == 0:
        splines.append(Spline(1139, 'inner', n_teeth=6, d=23 / 1_000, D=26 / 1_000))
        conditions.append({'moment': 40, 'length': 40 / 1_000, 'unevenness': 1.5})

    if 6033:
        splines.append(Spline(6033, 'inner', n_teeth=54, module=8 / 1_000, D=440 / 1_000))
        conditions.append({'moment': 40, 'length': 40 / 1_000, 'unevenness': 1.5})

    for spline, condition in zip(splines, conditions):
        print(spline)
        print(f'{spline.tension(**condition) = }')
        fitted_splines = Spline.fit(spline.standard, spline.join, 40 * 10 ** 6, 20, 20 / 1000, 1)
        for fs in fitted_splines: print(fs)

        spline = Spline(spline.standard, spline.join, **fitted_splines[0])
        spline.show()


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
