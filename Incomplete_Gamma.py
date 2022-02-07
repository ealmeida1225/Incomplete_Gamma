# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy.stats import gamma
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats


class Incomplete_Gamma:
    def __init__(self, data):
        self.data = pd.Series(data)
        # self.sorted_pd_list = pd.Series(sorted(self.original_list))
        self.ln_list = self.data.apply(lambda x: 0 if x == 0 else np.log(x))
        self.suma_ln_list = self.ln_list.sum()

        # total de registros
        self.N = len(self.data)
        # cant de registros mayores que 0
        self.n = self.data[self.data.iloc[:] > 0].count()
        # cantidad de días sin lluvia
        self.No = self.N - self.n
        # Total de precipitaciones en los datos
        self.sum_precip = self.data.sum()
        # promedio de las precipitaciones mayores que 0
        self.Xm = self.data[self.data.iloc[:] > 0].mean()
        # LN del promedio de las precipitaciones mayores que 0
        self.LN_Xm = np.log(self.Xm)

        self.A = self.LN_Xm - (self.suma_ln_list / self.n)
        self.alpha = (1 / (4 * self.A)) * (pow((4 * self.A) / 3, 0.5) + 1)
        self.beta = self.Xm / self.alpha
        # probabilidad de días con precipitación 0
        self.q = self.No / self.N
        # probabilidad de días con precipitación diferente de 0
        self.p = 1 - self.q

        self.H_x_values = list(np.arange(0, 1, 0.05))
        while self.q > self.H_x_values.pop(0):
            pass
        self.df = pd.DataFrame(self.H_x_values, columns=(['H(x)']))
        self.df['G(x)'] = self.df.apply(lambda x: (x - self.q) / self.p)
        self.df['T(años)'] = self.df['G(x)'].apply(
            lambda x: 1 / (1 - x))
        self.df['P(mm)'] = self.df['G(x)'].apply(
            lambda x: gamma.ppf(x, self.alpha, 0, self.beta))  # Percent point function (inverse of cdf — percentiles).
        self.params, self.cov = self.adjust_curve_from_data()

    def my_fit_function(self, x, a, b):
        """
        Función logarítmica que caracteriza las precipitaciones en función del tiempo
        Args:
            x: valores experimentales
            a: valor a ajustar
            b: valor a ajustar
        """
        return a * np.log(x) + b

    def adjust_curve_from_data(self):
        params, cov = curve_fit(f=self.my_fit_function,
                                xdata=self.df['T(años)'],
                                ydata=self.df['P(mm)'],
                                bounds=(-np.inf, np.inf))
        return params, cov

    def plot_function(self):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
        axes.plot(self.df['T(años)'], self.df['P(mm)'], 'b.', label="Datos")

        xnew = np.linspace(1, 16, 50)
        axes.plot(xnew, self.my_fit_function(xnew, *self.params), 'g', label="Función estimada")
        axes.set_xlabel('T(años)')
        axes.set_ylabel('P(mm)')
        axes.set_title('Estimación usando la distribución Gamma Incompleta')

        function = ('{0}*ln(x)+({1})'.format(round(self.params[0], 3),
                                             round(self.params[1], 3)))

        y_original = self.df['P(mm)']
        y_stimated = self.my_fit_function(self.df['T(años)'], *self.params)
        at = AnchoredText('f(x)={0}\n R^2 = {1}'
                          .format(function, round(pow(np.corrcoef(y_original, y_stimated)[0, 1], 2), 4)),
                          prop=dict(size=10, color='m'), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.3")
        axes.add_artist(at)

        axes.legend()
        fig.show()

    def get_adjust_function(self):
        return '{0}*ln(x)+{1}'.format(self.params[0], self.params[1])

original_list = pd.Series(
    [0.00, 0.00, 9.40, 6.00, 0.00, 10.50, 44.80, 8.70, 10.00, 2.80, 12.30, 3.90, 3.20, 2.50, 8.20, 37.00, 16.70, 4.80,
     2.80, 68.60, 71.20, 9.70, 72.90, 13.80, 0.00, 5.60, 4.00, 37.90, 2.60, 0.00, 0.00])

incomplete_gamma = Incomplete_Gamma(original_list)
incomplete_gamma.plot_function()
