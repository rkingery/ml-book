import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

local = Path().cwd().parent / 'local'

## basic-math

def plot_function(x, f, xlim=None, ylim=None, title=None, show_grid=True):
    xlow, xhigh = xlim if xlim is not None else (-10, 10)
    ylow, yhigh = ylim if ylim is not None else xlim
    plt.figure(figsize=(4, 3))
    plt.hlines(0 * x, xlow, xhigh, color='black', linewidth=0.7)
    if not isinstance(f, tuple) or isinstance(f, list):
        plt.vlines(0 * f(x), ylow, yhigh, color='black', linewidth=0.7)
        plt.plot(x, f(x), color='red')
    else:
        plt.vlines(0 * f[0](x), ylow, yhigh, color='black', linewidth=0.7)
        for fn in f:
            plt.plot(x, fn(x))
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    if show_grid:
        plt.grid(True, alpha=0.5)
        plt.xticks(range(int(xlow), int(xhigh) + 1, 1))
        plt.yticks(range(int(ylow), int(yhigh) + 1, 1))
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(*xlim)
    plt.show();
    
def query_wolfram_alpha(query, api_file='wolfram_key.txt', answer='formatted'):
    import wolframalpha
    api_key = (local / api_file).read_text()
    client = wolframalpha.Client(api_key)
    response = client.query(query)
    if answer == 'formatted':
        answer = next(response.results).text
    return answer

def plot_3d(x, y, f, title='', show_ticks=True, elev=30, azim=30):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('x', labelpad=-10)
        ax.set_ylabel('y', labelpad=-10)
        ax.set_zlabel('z', labelpad=-10)
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
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

    
## numerical-computing

def gen_all_floats(n_precision, n_exp, bias):
    n_precision -= 1  # last bit is used for reserved numbers
    exp_min, exp_max = 1 - bias, 2 ** n_exp - 1 - bias
    x = []
    for m in range(exp_min, exp_max + 1):
        max_val = 2 ** n_precision - 1
        for n in range(max_val + 1):
            precision = 1 + n / (2 ** n_precision)
            for sign in [-1, 1]:
                num = sign * precision * 2 ** m  # definition of float
                x.append(num)
    return sorted(x)

def plot_number_dist(x, title=''):
    plt.figure(figsize=(12, 1))
    plt.scatter(x, np.ones(len(x)), c='red', s=4);
    plt.title(title)
    plt.yticks([])
    plt.title(title)
    plt.show()
    

## linear-algebra

def plot_vectors(vs, xlim=(), ylim=(), title='', labels=None):
    plt.figure(figsize=(4, 3))
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
    plt.figure(figsize=(4, 3))
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
    plt.figure(figsize=(4, 3))
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
    

## calculus

def plot_right_triangle(points=[(0, 0), (1, 0), (1, 1)], base_label='$dx$', height_label='$dy$', 
                        hyp_label='slope$=dy/dx$', offset=0.08):
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    x = [x0, x1, x2]
    y = [y0, y1, y2]
    hyp_x = [x[0], x[2]]
    hyp_y = [y[0], y[2]]
    mid_base = (x[1] - x[0]) / 2 + x[0]
    mid_height = (y[2] - y[1]) / 2 + y[1]
    plt.figure(figsize=(4, 2.66))
    plt.plot(x, y, color='black')
    plt.plot(hyp_x, hyp_y, color='black')
    plt.text(mid_base, y[0] - 1.5 * offset, base_label)
    plt.text(x[1] + 0.5 * offset, mid_height, height_label)
    plt.text(mid_base - 2.5 * offset, mid_height + 2.5 * offset, hyp_label)
    plt.axis('off')
    plt.show()

def plot_tangent_plane(x, y, x0, y0, f, f_tangent, title=''):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z_tangent = f_tangent(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.4)
    ax.plot_surface(X, Y, Z_tangent, color='green', alpha=0.4)
    ax.scatter([x0], [y0], [f(x0, y0)], s=50, marker='.', zorder=10, color='red')
    # ax.view_init(elev=40, azim=90)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.show()
    
def plot_tangent_contour(x, y, x0, y0, f, f_tangent, dfdx, title=''):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z0 = f_tangent(X, Y)
    
    grad = dfdx(x0, y0)
    m = - grad[1] / grad[0]
    b = y0 - m * x0
    y_tangent = m * x + b

    plt.figure(figsize=(4, 3))
    plt.contour(X, Y, Z)
    plt.plot(x, y_tangent, color='red')
    plt.scatter([x0], [y0], marker='o', color='red')
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.show()
    
def plot_area_under_curve(x, f, dx=1, show_all_xticks=True):
    y = f(x)
    x_rect = np.arange(min(x), max(x), dx) + dx / 2
    n_rects = len(x_rect)
    y_rect = f(x_rect + dx / 2)
    print(f'Approximate Area: {np.sum(y_rect * dx)}')
    plt.figure(figsize=(4, 3))
    plt.plot(x, y, color='red')
    plt.bar(x_rect, y_rect, width=dx, alpha=1, edgecolor='black', facecolor='none', linewidth=1)
    plt.title(f'{n_rects} Rectangles')
    if show_all_xticks:
        plt.xticks(np.arange(min(x), max(x) + 1))
    plt.show()
    

# probability

def print_table(data, columns):
    data = np.array(data).T
    df = pd.DataFrame(data=data, columns=columns)
    print(df.to_string(index=False))
    
def plot_3d_hist(x, y, bins=10, xlim=(-2, 2), ylim=(-2, 2), elev=20, azim=30, title='', show_ticks=False):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.view_init(elev=elev, azim=azim)
    #ax.set_axis_off()
    ax.set_title(title)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('x', labelpad=-10)
        ax.set_ylabel('y', labelpad=-10)
        ax.set_zlabel('counts', labelpad=-10)
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('counts')
    plt.show()
    
def plot_multivariate_gaussian(mu, Sigma, show_ticks=False, elev=30, azim=30):
    from scipy.stats import multivariate_normal
    from matplotlib import gridspec
    dist = multivariate_normal(mu, Sigma)
    p = lambda X: dist.pdf(X)
    lim = max(1.5 * Sigma[0, 0], 1.5 * Sigma[1, 1])
    lim = (-lim, lim)
    x = np.linspace(lim[0], lim[1], 1000)
    y = np.linspace(lim[0], lim[1], 1000)
    x, y = np.meshgrid(x, y)
    z = p(np.dstack((x, y)))
    fig = plt.figure(figsize=(12,5))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1.5, 1], wspace=1, hspace=1)#, height_ratios=[2, 1])
    ax1 = fig.add_subplot(spec[0], projection='3d')
    ax1.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax1.view_init(elev=elev, azim=azim)
    fig.suptitle('Multivariate Gaussian $\mathcal{N}(\mu,\Sigma)$\n' 
                 + f'$\mu=${mu.tolist()} \n $\Sigma=${Sigma.tolist()}', y=0.9, fontsize=11)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_zticks([])
    ax1.set_zlabel('p', labelpad=-10)
    if not show_ticks:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel('x', labelpad=-10)
        ax1.set_ylabel('y', labelpad=-10)
    else:
        ax1.tick_params(axis='y',direction='in', pad=-4)
        ax1.tick_params(axis='x',direction='in', pad=-4)
        ax1.set_xticks(np.arange(xlim[0], xlim[1] + 1))
        ax1.set_yticks(np.arange(ylim[0], ylim[1] + 1))
        ax1.set_xlabel('x', labelpad=-5)
        ax1.set_ylabel('y', labelpad=-5)
    ax1.set_title('3D Plot', y=0.9)
    ax2 = fig.add_subplot(spec[1])
    ax2.contour(x, y, z)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    lim = max(1.5 * Sigma[0, 0], 1.5 * Sigma[1, 1])
    lim = (-lim, lim)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_title('Contour Plot', y=1)
    fig.subplots_adjust(wspace=1)
    plt.show()
    

# statistics

def plot_gradient_descent(f, grad_fn, x0, alpha, n_iters, annotate_start_end=True, xlim=None, ylim=None, title=''):
    def gradient_descent(f, grad_fn, x0, alpha, n_iters):
        points = [(x0, f(x0))]
        for i in range(n_iters):
            x0 = x0 - alpha * grad_fn(x0)
            points.append((x0, f(x0)))
        return points
    
    points = gradient_descent(f, grad_fn, x0, alpha, n_iters)
    x0, y0 = zip(*points)
    x0, y0 = np.array(x0), np.array(y0)
    lim = np.max([np.abs(np.min(x0)), np.abs(np.max(x0))])
    x = np.linspace(-1.5 * lim, 1.5 * lim, 100)
    y = f(x)
    plt.figure(figsize=(5, 4))
    plt.plot(x, f(x), color='black')
    plt.scatter(x0, y0, color='red', s=10)
    if annotate_start_end:
        plt.annotate('start', points[0], xytext=(-10, 5), textcoords='offset points')
        plt.annotate('end', points[-1], xytext=(-10, 5), textcoords='offset points')
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        plt.plot([x1, x2], [y1, y2], color='red')
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()