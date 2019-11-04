import copy

import cexprtk
import numpy as np
import scipy as sp


class ExpressionWrapper:
    def __init__(self, math_str, parameters):
        symbols = {}
        for param in parameters:
            symbols[param] = 1.0
        st = cexprtk.Symbol_Table(symbols, add_constants=True)
        self.expression = cexprtk.Expression(math_str, st)
        self._constant = 1

    def _evaluate_for_parameters(self, parameters):
        for key in parameters:
            self.expression.symbol_table.variables[key] = parameters[key]
        return self._constant * self.expression()

    def _evaluate_for_x(self, x):
        return self._evaluate_for_parameters({'x': x})

    def _evaluate_with_list(self, values):
        result = []
        for x in values:
            result.append(self._evaluate_for_x(x))
        return result

    def _get_minus_func(self):
        minus_func = copy.copy(self)
        minus_func._constant = -1
        return minus_func

    def get_minimum_for_partition(self, p):
        return [sp.optimize.minimize_scalar(self, bounds=(p[i], p[i + 1]), method='bounded').fun for i in range(len(p)-1)]

    def get_maximum_for_partition(self, p):
        minus_func = self._get_minus_func()
        min = minus_func.get_minimum_for_partition(p)
        return [-x for x in min]

    def __call__(self, values):
        if type(values) is np.ndarray or type(values) is list:
            return self._evaluate_with_list(values)
        elif type(values) is np.float64 or type(values) is float:
            return self._evaluate_for_x(values)
        else:
            return self._evaluate_for_parameters(values)
