from graph import plot
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from graph import trace_values, make_subplots

def term_output(term, input_value):
    return term[0]*input_value**term[1]

def output_at(list_of_terms, x_value):
    outputs = list(map(lambda term: term_output(term, x_value), list_of_terms))
    return sum(outputs)

def delta_f(list_of_terms, x_value, delta_x):
    return output_at(list_of_terms, x_value + delta_x) - output_at(list_of_terms, x_value)

def delta_f_trace(list_of_terms, x_value, delta_x):
    initial_f = output_at(list_of_terms, x_value)
    delta_y = delta_f(list_of_terms, x_value, delta_x)
    trace = trace_values(x_values=[x_value + delta_x, x_value + delta_x],
                        y_values=[initial_f, initial_f + delta_y],
                        text=[str(initial_f), str(initial_f + delta_y)], mode = 'lines+text', name = 'y2 - y1 = ' + str(initial_f + delta_y) + ' - '  + str(initial_f) + ' = ' + str(delta_y),
                        options = {'textposition': 'right'})
    return trace

def delta_x_trace(list_of_terms, x_value, delta):
    initial_f = output_at(list_of_terms, x_value)
    trace = trace_values(x_values=[x_value, x_value + delta],
                         text=[str(x_value), str(x_value + delta)],
                         y_values=[initial_f, initial_f], mode = 'lines+text',
                         name = 'x2 - x1 = ' + str(initial_f + delta) + ' - '  + str(initial_f) + ' = ' + str(delta),
                         options = {'textposition': 'bottom'})
    return trace

def derivative_of(list_of_terms, x_value, delta_x):
    delta = delta_f(list_of_terms, x_value, delta_x)
    return round(delta/delta_x, 3)

def derivative_trace(list_of_terms, x_value, line_length = 4, delta = .01):
    derivative_at = derivative_of(list_of_terms, x_value, delta)
    y = output_at(list_of_terms, x_value)
    x_minus = x_value - line_length/2
    x_plus = x_value + line_length/2
    y_minus = y - derivative_at * line_length/2
    y_plus = y + derivative_at * line_length/2
    trace = {'x': [x_minus, x_value, x_plus],
    'y': [y_minus, y, y_plus],
    'text': ['', "f' = " +  str(derivative_at), ''],
    'mode': 'text+lines', 'textposition': 'bottom'}
    return trace

def delta_traces(list_of_terms, x_value, line_length = 4, delta_x = .01):
    tangent = derivative_trace(list_of_terms, x_value, line_length, delta_x)
    delta_f_line = delta_f_trace(list_of_terms, x_value, delta_x)
    delta_x_line = delta_x_trace(list_of_terms, x_value, delta_x)
    return [tangent, delta_f_line, delta_x_line]

def function_values_trace(list_of_terms, x_values):
    function_values = list(map(lambda x: output_at(list_of_terms, x),x_values))
    return trace_values(x_values, function_values, mode = 'line')

def derivative_values_trace(list_of_terms, x_values, delta_x):
    derivative_values = list(map(lambda x: derivative_of(list_of_terms, x, delta_x), x_values))
    return trace_values(x_values, derivative_values, mode = 'line')

def function_and_derivative_trace(list_of_terms, x_values, delta_x):
    traced_function = function_values_trace(list_of_terms, x_values)
    traced_derivative = derivative_values_trace(list_of_terms, x_values, delta_x)
    return make_subplots([traced_function], [traced_derivative])
