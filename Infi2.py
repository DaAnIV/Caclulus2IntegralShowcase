import base64

from flask import Flask, render_template, request, jsonify
import scipy as sp
import scipy.optimize
import scipy.integrate
import numpy as np
import cexprtk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import copy

app = Flask(__name__)


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


def _get_function_from_math_string(math_str, parameters=None):
    if parameters is None:
        parameters = ['x']
    return ExpressionWrapper(math_str, parameters)


def create_riemman_figure(func, P, T, dt, a, b, n):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    # Plot the function values at the chosen points, and the rectangles
    axis.plot(T, func(T), '.', markersize=10)
    axis.bar(P[:-1], func(T), width=dt, alpha=0.2, align='edge', edgecolor='black')

    # Now plot the "true" curve of the function
    x = np.linspace(a, b, n * 100)  # we take finer spacing to get a "smooth" graph
    y = func(x)
    axis.plot(x, y)
    axis.axis('off')
    return fig


def create_darboux_figure(func, P, dt, a, b, n):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    lower_darboux = func.get_minimum_for_partition(P)
    upper_darboux = func.get_maximum_for_partition(P)

    # Plot the function values at the chosen points, and the rectangles
    axis.bar(P[:-1], lower_darboux, width=dt, alpha=0.2, align='edge', linewidth=0.8, color='blue', edgecolor='blue')
    axis.bar(P[:-1], upper_darboux, width=dt, alpha=0.2, align='edge', linewidth=0.8, color='red', edgecolor='red')

    # Now plot the "true" curve of the function
    x = np.linspace(a, b, n * 100)  # we take finer spacing to get a "smooth" graph
    y = func(x)
    axis.plot(x, y)
    axis.axis('off')
    return fig


def get_encoded_image(fig):
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded_riemann_img = base64.b64encode(output.getvalue()).decode('utf-8').replace('\n', '')
    output.close()
    return encoded_riemann_img


@app.route("/")
def main():
    return render_template('index.html')


@app.route('/get_plot', methods=['POST'])
def get_plot():
    _function_txt = request.form['function']
    _error_method_txt = request.form['error_method']
    _a = np.float32(cexprtk.evaluate_expression(request.form['a'], {}))
    _b = np.float32(cexprtk.evaluate_expression(request.form['b'], {}))
    _n = np.int32(cexprtk.evaluate_expression(request.form['n'], {}))
    try:
        _function = _get_function_from_math_string(_function_txt)
    except cexprtk.ParseException as e:
        return jsonify({'error': "Parse error {}".format(e)})

    P, dt = np.linspace(_a, _b, _n, retstep=True)  # Standard partition constant width
    T = [np.random.rand() * dt + p for p in P[:-1]]  # Randomly chosen point

    riemann_fig = create_riemman_figure(_function, P, T, dt, _a, _b, _n)
    darboux_fig = create_darboux_figure(_function, P, dt, _a, _b, _n)

    # Run python's numerical integration function
    num_int, err = sp.integrate.quad(_function, _a, _b)  # err is an estimate of the error

    r_sum = sum([_function(t) * dt for t in T])
    diff = r_sum - num_int

    result = {
        'riemann_title': 'Riemann sum with n = {} points'.format(int(_n)),
        'riemann_img': '<img src="data:image/png;base64,{}" />'.format(get_encoded_image(riemann_fig)),
        'darboux_title': 'Upper and lower Darboux sums with n = {} points'.format(int(_n)),
        'darboux_img': '<img src="data:image/png;base64,{}" />'.format(get_encoded_image(darboux_fig)),
        'int_result': 'Numercial integration gives {} with possible error of {}'.format(num_int, err),
        'diff_result': 'The difference between our Riemann sum and the true value is {}'.format(diff)
    }

    if len(_error_method_txt) > 0:
        _error_method_function = _get_function_from_math_string(_error_method_txt, ['a', 'b', 'n'])
        _error_bound = _error_method_function({'a': _a, 'b': _b, 'n': _n})
        result['error_bound'] = 'The error absolute bound is evaluated as {}, diff between bound and actual error {}'.format(_error_bound, _error_bound - abs(diff))

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
