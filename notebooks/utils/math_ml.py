import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

local = Path().cwd().parent / 'local'

## basic-math

def plot_function(x, f, xlim=None, ylim=None, title=None, ticks_every=None, labels=None, xlabel='$x$', ylabel='$y$'):
    xlim = xlim if xlim is not None else (min(x), max(x))
    ylim = ylim if ylim is not None else xlim
    xlow, xhigh = xlim
    ylow, yhigh = ylim
    if not isinstance(f, list):
        f = [f]
    plt.figure(figsize=(4, 3))
    plt.hlines(0 * x, xlow, xhigh, color='black', linewidth=0.7)
    plt.vlines(0 * f[0](x), ylow, yhigh, color='black', linewidth=0.7)
    for i, fn in enumerate(f):
        label = labels[i] if isinstance(labels, list) else None
        plt.plot(x, fn(x), label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ticks_every is not None:
        plt.xticks(np.arange(int(xlow), int(xhigh) + 1, ticks_every))
        plt.yticks(np.arange(int(ylow), int(yhigh) + 1, ticks_every))
    plt.grid(True, alpha=0.5)
    if labels is not None:
        plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.show()
    
def query_wolfram_alpha(query, api_file='wolfram_key.txt', answer='formatted'):
    import wolframalpha
    api_key = (local / api_file).read_text()
    client = wolframalpha.Client(api_key)
    response = client.query(query)
    answer = next(response.results).texts
    return answer

def plot_3d(x, y, f, title='', show_ticks=True, elev=30, azim=30, labels=None):
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not isinstance(f, list):
        ax.plot_surface(X, Y, f(X, Y), label=labels)
    else:
        for i, fn in enumerate(f):
            label = labels[i] if isinstance(labels, list) else None
            ax.plot_surface(X, Y, fn(X, Y), label='hey', zorder=i)
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
    if labels is not None:
        pass # ax.legend()
    plt.show()
    
def plot_countour(x, y, f, title=''):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.figure(figsize=(4, 3))
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, fmt='%1.1f')
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

    
## numerical-computing

def represent_as_float(x, n, n_exp, n_man, bias):
    import math
    # get sign
    sign = 1 if x < 0 else 0
    x = -x if sign == 1 else x
    # get exponent and precision = 1 + mantissa
    exponent = 0 if x == 0 else int(math.log2(x))
    precision = 0 if x == 0 else x / (2 ** exponent)
    # normalize precision to be between 1 and 2
    if precision >= 2:
        precision /= 2
        exponent += 1
    elif precision < 1:
        precision *= 2
        exponent -= 1
    # bias exponent and convert to binary
    exp_biased = exponent + bias
    exp_bits = bin(exp_biased)[2:][:n_exp]
    exp_bits = exp_bits[:n_exp]
    if n_exp >= len(exp_bits):
        exp_bits = '0' * (n_exp - len(exp_bits)) + exp_bits
    # get mantissa and convert it to binary
    mantissa = precision - 1
    man_bits = bin(int(mantissa * 2 ** n_man))[2:]
    man_bits = man_bits[-n_man:]
    # print output
    print(f'scientific notation: (-1)^{sign} * (1 + {precision-1}) * 2^{exponent}')
    print(f'{n}-bit floating point representation: {sign} {exp_bits} {man_bits}')

def gen_all_floats(n, n_man, n_exp, bias):
    n_man -= 1  # last bit is used for reserved numbers
    exp_min, exp_max = 1 - bias, 2 ** n_exp - 1 - bias
    x = []
    for exp in range(exp_min, exp_max + 1):
        max_val = 2 ** n_man - 1
        for n in range(max_val + 1):
            mantissa = n * 2 ** (-n_man)
            for sign in [-1, 1]:
                num = sign * (1 + mantissa) * 2 ** exp
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

def plot_vectors(vectors, xlim=None, ylim=None, title='', labels=None, colors=None, tails=None, text_offsets=None, 
                 zorders=None, ticks_every=None, **kwargs):
    if not isinstance(vectors, list):
        vectors = [vectors]
    if not isinstance(tails, list):
        tails = [[0, 0] for _ in range(len(vectors))]
    all_x = [v[0] for v in vectors] + [tail[0] for tail in tails]
    all_y = [v[1] for v in vectors] + [tail[1] for tail in tails]
    xlim = (min(all_x) - 1, max(all_x) + 1) if xlim is None else xlim
    ylim = (min(all_y) - 1, max(all_y) + 1) if ylim is None else ylim
    xlow, xhigh = xlim
    ylow, yhigh = ylim
    plt.figure(figsize=(4, 3))
    plt.hlines(0 * np.arange(xlow, xhigh, 100), xlow, xhigh, color='black', linewidth=0.5)
    plt.vlines(0 * np.arange(ylow, yhigh, 100), ylow, yhigh, color='black', linewidth=0.5)
    for i, v in enumerate(vectors):
        zorder = zorders[i] if isinstance(zorders, list) else 10 - i
        label = labels[i] if isinstance(labels, list) else None
        color = colors[i] if isinstance(colors, list) else 'red'
        text_offset = np.array(text_offsets[i]) if isinstance(text_offsets, list) else np.array([0, 0])
        tail = np.array(tails[i])
        plt.quiver(tail[0], tail[1], v[0], v[1], scale=1, angles='xy', scale_units='xy', zorder=zorder, color=color, **kwargs)
        text_loc = v + text_offset if isinstance(text_offsets, list) else v
        plt.annotate(label, v + tail, fontsize=15, zorder=zorder, xytext=(text_loc[0].item(), text_loc[1].item()))
    if ticks_every is not None:
        plt.xticks(np.arange(int(xlow), int(xhigh) + 1, ticks_every))
        plt.yticks(np.arange(int(ylow), int(yhigh) + 1, ticks_every))
    plt.grid(True, alpha=0.5)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title(title)
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

def plot_tangent_curve(x, x0, f, f_tangent, xlim=None, ylim=None, title=None):
    plt.figure(figsize=(4, 3))
    plt.plot(x, f(x))
    plt.plot(x, f_tangent(x))
    plt.scatter([x0], [f(x0)], marker='o', color='red')
    plt.title(title)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.grid(True, alpha=0.5)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.show()
    
def plot_tangent_plane(x, y, x0, y0, f, f_tangent, dfdx, title='', plot_grad=False, grad_scale=2):
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)

    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)
    setattr(Axes3D, 'arrow3D', _arrow3D)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    Z_tangent = f_tangent(X, Y)
    grad = dfdx(x0, y0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.4, zorder=1)
    ax.plot_surface(X, Y, Z_tangent, color='green', alpha=0.4, zorder=2)
    if plot_grad:
        offset = 0.01
        ax.arrow3D(x0 + offset, y0 + offset, f(x0, y0) + offset, grad[0] / grad_scale, grad[1] / grad_scale, 0, 
                   mutation_scale=15, fc='red')
    ax.scatter([x0], [y0], [f(x0, y0)], s=250, marker='.', color='red', zorder=4, edgecolors='black')
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

    plt.figure(figsize=(4, 4))
    plt.contour(X, Y, Z, zorder=1)
    plt.plot(x, y_tangent, color='green', zorder=2, linewidth=3)
    plt.quiver(x0, y0, grad[0], grad[1], color='red', angles='xy', scale_units='xy', scale=3, headwidth=3,
               zorder=3, width=0.013, headlength=4, edgecolors='black', linewidth=1)
    plt.scatter([x0], [y0], marker='o', color='red', edgecolors='black', zorder=4)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.show()
    
def plot_approximating_rectangles(x, f, dx=1, show_all_xticks=True):
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
    
def plot_function_with_area(x, f, a, b, title='', xlabel='$x$', ylabel='$y$', **kwargs):
    y = f(x)
    mask = (x >= a) & (x <= b)
    plt.plot(x, y, color='red', **kwargs)
    plt.fill_between(x, y, where=mask)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.5)
    plt.show()
    
    
# probability
    
def plot_histogram(x, is_discrete=False, title='', **kwargs):
    if is_discrete:
        sns.histplot(x, discrete=True, shrink=0.8, **kwargs)
        unique = np.unique(x)
        if len(unique) < 15:
            plt.xticks(unique)
    else:
        sns.histplot(x, **kwargs)
    plt.title(title)
    plt.show()
    
def plot_joint_histogram(X, title='', figsize=(4, 3)):
    X = np.column_stack(X)
    counter = np.unique([str(tuple(X[i])) for i in range(len(X))], return_counts=True)
    plt.figure(figsize=figsize)
    sns.barplot(x=counter[0], y=counter[1], color=sns.color_palette()[0], edgecolor='black', linewidth=1.3, alpha=0.75)
    plt.xticks(rotation=90)
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    
def plot_hist2d(x, y, title='', figsize=(4, 3), bins=(10, 10)):
    plt.figure(figsize=figsize)
    plt.hist2d(x, y, bins=bins, cmap='Blues')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title(title)
    plt.show()

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
    

# optimization

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