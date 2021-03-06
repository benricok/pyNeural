import matplotlib.pyplot as plt
import numpy as np


# def f(x):
#    return 2*x**2


def f(x):
    return 5*x**3 + 2*x**2 - 5*x + 7


x = np.arange(-10, 10, 0.001)
y = f(x)

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c', 'k', 'g', 'r', 'b', 'c']


def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative*x) + b


for i in np.arange(0, 10, 2):
    # The point and the "close enough" point
    p2_delta = 0.0001
    x1 = i
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))

    # Derivative approximation and y-intercept for the tangent line
    approximate_derivative = (y2-y1)/(x2-x1)
    b = y2 - approximate_derivative*x2
    toPlot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[abs(i)])
    plt.plot([point for point in toPlot],
             [approximate_tangent_line(point, approximate_derivative)
              for point in toPlot],
             c=colors[abs(i)])

    print('Approximate derivative fot f(x)',
          f'where x = {x1} is {approximate_derivative}')

plt.show()

# https://nnfs.io/but
# https://nnfs.io/ch7
