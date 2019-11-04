import base64

from flask import Blueprint, render_template, jsonify
import scipy as sp
import scipy.integrate
import numpy as np
import cexprtk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask_wtf import FlaskForm
from wtforms.fields import *
import wtforms.validators as validators

from expression_wrapper import ExpressionWrapper

integral_blueprint = Blueprint('integrals', __name__)


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


class PlotForm(FlaskForm):
    function_text = StringField(u"Function to plot", [validators.required()])
    a = FloatField(u"Start", default=0)
    b = FloatField(u"End", default=1)
    n = IntegerField(u"Partition count", [validators.number_range(min=1)], default=20)
    error_method_text = StringField(u"Error method", [validators.optional()])
    submit = SubmitField("Plot")


@integral_blueprint.route("/")
def infi_integral():
    return render_template('index.html', form=PlotForm())


@integral_blueprint.route('/get_plot', methods=['POST'])
def get_plot_post():
    form = PlotForm()
    if form.validate_on_submit():
        return get_plot(form.function_text.data, form.error_method_text.data, np.float32(form.a.data),
                        np.float32(form.b.data), np.int32(form.n.data))
    return jsonify(data=form.errors)


def get_plot(function_txt, error_method_txt, a, b, n):
    try:
        _function = _get_function_from_math_string(function_txt)
    except cexprtk.ParseException as e:
        return jsonify({'error': "Parse error {}".format(e)})

    P, dt = np.linspace(a, b, n+1, retstep=True)  # Standard partition constant width
    T = [np.random.rand() * dt + p for p in P[:-1]]  # Randomly chosen point

    riemann_fig = create_riemman_figure(_function, P, T, dt, a, b, n)
    darboux_fig = create_darboux_figure(_function, P, dt, a, b, n)

    # Run python's numerical integration function
    num_int, err = sp.integrate.quad(_function, a, b)  # err is an estimate of the error

    r_sum = sum([_function(t) * dt for t in T])
    diff = r_sum - num_int

    result = {
        'riemann_title': 'Riemann sum with n = {} points'.format(int(n)),
        'riemann_img': '<img src="data:image/png;base64,{}" />'.format(get_encoded_image(riemann_fig)),
        'darboux_title': 'Upper and lower Darboux sums with n = {} points'.format(int(n)),
        'darboux_img': '<img src="data:image/png;base64,{}" />'.format(get_encoded_image(darboux_fig)),
        'int_result': 'Numercial integration gives {} with possible error of {}'.format(num_int, err),
        'diff_result': 'The difference between our Riemann sum and the true value is {}'.format(diff)
    }

    if len(error_method_txt) > 0:
        _error_method_function = _get_function_from_math_string(error_method_txt, ['a', 'b', 'n'])
        _error_bound = _error_method_function({'a': a, 'b': b, 'n': n})
        result['error_bound'] = 'The error absolute bound is evaluated as {}, diff between bound and actual error {}'.format(_error_bound, _error_bound - abs(diff))

    return jsonify(result)
