import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

local = Path().cwd().parent / 'local'

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