import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

local = Path().cwd().parent / 'local'

## basic-math

def gen_all_floats(n_mantissa, n_exp, bias):
    n_mantissa -= 1
    n_digits = int(np.log(2 ** n_mantissa) / np.log(10)) + 1
    exp_min, exp_max = 1 - bias, 2 ** n_exp - 1 - bias
    x = []
    for m in range(exp_min, exp_max + 1):
        max_val = 2 ** n_mantissa - 1
        for n in range(max_val + 1):
            mantissa = 1 + n / (2 ** n_mantissa)
            for sign in [-1, 1]:
                num = sign * mantissa * 2 ** m
                num = np.round(num, n_digits)
                x.append(num)
    return sorted(x)

def plot_float_dist(x, title=''):
    plt.figure(figsize=(12, 1))
    plt.scatter(x, np.ones(len(x)), c='red', s=4);
    plt.title('Distribution of All Floats')
    plt.yticks([])
    plt.title(title)
    plt.show()

def plot_function(x, y, xlim, ylim=None, title=None):
    xlow, xhigh = xlim
    ylow, yhigh = ylim if ylim is not None else xlim
    plt.figure(figsize=(4, 3))
    plt.hlines(0 * x, xlow, xhigh, color='black', linewidth=0.7)
    if not isinstance(y, tuple):
        plt.vlines(0 * y, ylow, yhigh, color='black', linewidth=0.7)
        plt.plot(x, y, color='red')
    else:
        plt.vlines(0 * y[0], ylow, yhigh, color='black', linewidth=0.7)
        for yy in y:
            plt.plot(x, yy)
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True, alpha=0.5)
    plt.xticks(range(int(xlow), int(xhigh) + 1, 1))
    plt.yticks(range(int(ylow), int(yhigh) + 1, 1))
    plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(*xlim)
    plt.show();
    
def query_wolfram_alpha(query, api_file='wolfram_key.txt'):
    import wolframalpha
    api_key = (local / api_file).read_text()
    client = wolframalpha.Client(api_key)
    response = client.query(query)
    answer = next(response.results).text
    return answer

def plot_3d(x, y, f, title=''):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.show()
    
def plot_countour(x, y, f, title=''):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, fmt='%1.1f')
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

## linear-algebra

def plot_vectors(vs, xlim=(), ylim=(), title='', labels=None):
    plt.figure(figsize=(5, 4))
    if not isinstance(vs, list):
        vs = [vs]
    if labels is None:
        labels = ['v'] * len(vs)
    for i, v in enumerate(vs):
        plt.quiver(0, 0, v[0], v[1], scale=1, angles='xy', scale_units='xy', color='red')
        plt.annotate(labels[i], v, fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title)
    plt.show();

def plot_scalar_mult(v, cs, include_neg=False, xlim=(0, 3), ylim=(0, 3)):
    plt.figure(figsize=(5, 4))
    if not isinstance(cs, list):
        cs = [cs]
    for c in cs:
        plt.quiver(0, 0, c * v[0], c * v[1], angles='xy', scale_units='xy', scale=1, color='orange', 
                   label=f'${int(c)}v$', linewidth=2, edgecolors='orange', alpha=1)
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='black', label='$v$',
               linewidth=0.5, edgecolors='black', headwidth=5)
    if include_neg:
        plt.quiver(0, 0, -v[0], -v[1], angles='xy', scale_units='xy', scale=1, color='lime', label='$-v$',
                  linewidth=1, edgecolors='lime', headwidth=5)
    plt.grid(True, alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc='upper left')
    plt.title('Scalar Multiplication')
    plt.show();
    
def plot_vector_add(v, w, xlim=(0, 5), ylim=(0, 5)):
    plt.figure(figsize=(5, 4))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='$v$')
    plt.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1, color='green', label='$w$')
    plt.quiver(0, 0, v[0] + w[0], v[1] + w[1], angles='xy', scale_units='xy', scale=1, color='red', label='$v+w$')
    plt.grid(True, alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.title('Vector Addition')
    plt.show();