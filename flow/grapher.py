import math
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import os, os.path as opath
from re import match as rematch
from scipy.optimize import curve_fit
import statsmodels.api as sm

# =============================================================================================== #
# Ancillary useful functions

def color(name='gray'):
    """
    Return a color from a curated list of colors. Can take a name, an
    integer, or can pass through an RGB[A] or hex color. Default is
    gray.
    """

    colors = {
        'orange': '#E86E0A',
        'red': '#D61E21',
        'gray': '#7C7C7C',
        'black': '#000000',
        'green': '#75D977',
        'mint': '#47D1A8',
        'purple': '#C880D1',
        'indigo': '#5E5AE6',
        'blue': '#47AEED', # previously 4087DD
        'yellow': '#F2E205',
    }

    if isinstance(name, int):
        clrs = [colors[key] for key in colors]
        return clrs[name%len(clrs)]
    elif name in colors: return colors[name]
    elif rematch('^(#[0-9A-Fa-f]{6})|(rgb(a){0,1}\([0-9,]+\))$', name): return name
    else: return colors['gray']


def simplify(x, y, width=0):
        """
        Simplify a trace so that it is visually correct, but is possible
        to include in a figure for a paper. X is optional.
        """

        # Take care of optional arguments
        if isinstance(y, int) or isinstance(y, float):
            width = y
            y = x
            x = []

        # Take care of xs if necessary
        if len(x) == 0:
            x = np.arange(len(y))

        # Set the step size and return if already simple enough
        step = int(math.floor(len(y)/width))
        if step < 2:
            return (np.array(x), np.array(y))

        # Prepare for indexing
        x = np.array(x)
        y = np.array(y)

        newx = []
        newy = []
        for i in range(0, len(y), step):
            sub = y[i:i+step]
            mni = np.argmin(sub)
            mxi = np.argmax(sub)
            if mni < mxi:
                newx.extend([x[i+mni], x[i+mxi]])
                newy.extend([sub[mni], sub[mxi]])
            else:
                newx.extend([x[i+mxi], x[i+mni]])
                newy.extend([sub[mxi], sub[mni]])

        newx.append(x[-1])
        newy.append(y[-1])
        return (np.array(newx), np.array(newy))


def setargs(defaults, args):
        """
        Add input arguments to the default arguments. Confirm that
        arguments are the correct types.
        """

        out = {}
        # Set defaults of the argument
        for key in defaults:
            if type(defaults[key]) == list:
                if len(defaults[key]) > 0:
                    out[key] = defaults[key][0]
                else:
                    out[key] = []
            else: out[key] = defaults[key]

        # Add the input arguments and check them for correction
        for key in args:
            if key in defaults:
                if type(defaults) == list:
                    if args[key].lower() in defaults[key]:
                        out[key] = args[key].lower()
                else:
                    out[key] = args[key]
        return out

def scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    xtremes = ax.get_xlim()
    ytremes = ax.get_ylim()

    kwargs['sizex'] = pretty_scale(xtremes[1] - xtremes[0])
    kwargs['sizey'] = pretty_scale(ytremes[1] - ytremes[0])

    kwargs['labelx'] = str(kwargs['sizex'])
    kwargs['labely'] = str(kwargs['sizey'])

    #sb = AnchoredScaleBar(ax.transData, **kwargs)
    #ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    xran = xtremes[1] - xtremes[0]
    yran = ytremes[1] - ytremes[0]
    sizex, namex = pretty_scale(xran)
    sizey, namey = pretty_scale(yran)

    ax.axhline(ytremes[0], 1.0 - sizex/xran, 1.0, linewidth=4, color=color('gray'))
    ax.axvline(xtremes[1], 0.0, sizey/yran, linewidth=4, color=color('gray'))

    xunits = kwargs['xunits'] if 'xunits' in kwargs else ''
    yunits = kwargs['yunits'] if 'yunits' in kwargs else ''

    ax.text(0.96, 0.04, namex + xunits, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.text(0.97, 0.06, namey + yunits, horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, rotation=90)

    #plt.axvline(1.0 - sizex/xran, 1.0)

    #return sb

def pretty_scale(full, optimal_width=8, max_width=6):
    aim = full/optimal_width
    tens = math.floor(math.log10(aim))

    options = [i*math.pow(10, tens) for i in [1, 2, 3, 5, 10]]
    dists = sorted([(abs(aim - i), i) for i in options])

    if dists[0][1] < full/max_width:
        return dists[0][1], '%g' % (dists[0][1])
    else:
        return dists[1][1], '%g' % (dists[1][1])

def fitdecay(x, y):
    def efun(x, A, K, c): return A*np.exp(K*x) + c

    opt_parms, parm_cov = curve_fit(efun, x, y, maxfev=1000)
    lmx = np.unique(x)
    lmy = efun(lmx, opt_parms[0], opt_parms[1], opt_parms[2])
    return lmx, lmy

def fit_GLM(x, y, family='Gamma', link='log'):
    """
    Fit a nonlinear regression to the data
    :param x: vector of x values
    :param y: vector of y values
    :param family: family of values, string, can be gamma or poission
    :param link: link function, str, can be log
    :return: x, y of predictions
    """

    # Clean up values first
    allzeros = np.bitwise_and(x == 0, y == 0)
    nonnans = np.bitwise_and(np.isfinite(x), np.isfinite(y))
    nonnans = np.bitwise_and(nonnans, np.invert(allzeros))
    x, y = x[nonnans], y[nonnans]

    X = sm.add_constant(x)
    family = eval(
        'sm.families.{}(link=sm.families.links.{})'.format(family, link))
    glm = sm.GLM(y, X, family=family)
    glm_results = glm.fit()
    exog = glm.exog
    # n_x_vals = max(100, exog.shape[0])  # does not work because it's assigned back to exog
    n_x_vals = exog.shape[0]
    x_val = np.linspace(
        exog[:, 1].min(), exog[:, 1].max(), n_x_vals)
    exog[:, 1] = x_val
    predict = glm_results.get_prediction(exog)

    return x_val, predict.predicted_mean, predict.conf_int()[:, 0], predict.conf_int()[:, 1]


# =============================================================================================== #
# Class necessary for suppressing errors

# Define a context manager to suppress stdout and stderr.
class SuppressErrors(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

# =============================================================================================== #
# Graphing class

class Grapher():
    def __init__(self, save_directory='', width='full'):
        """
        Optionally set the save directory and width. Width can be full,
        half, or an integer.
        """

        self.dir(save_directory)
        self.size(width)
        self._data = []
        self._shaded = []

    def dir(self, save_directory=''):
        """
        Set the save directory. Default is the current directory.
        """

        self._directory = save_directory
        return self

    def size(self, width='full', height='full'):
        """
        Set the size of the output image. Width can be full, half, or an
        integer. Height can be full or an integer.
        """

        if isinstance(width, str):
            w = 7.5 if width == 'half' else 14.35
        else:
            w = width

        if isinstance(height, str):
            h = 6
        else:
            h = height

        self._size = (w, h)
        return self

    def add(self, x, y=[], **kwargs):
        """
        Add a set of data points. Optionally combine like xs.
        """

        defaults = {
            'errors': [],
            'name': '',
            'color': 'gray',
            'color2': '',
            'stroke': '#CCCCCC',
            'skip-zeros': False, # Skip zero values for averaging and plotting
            'skip-minus-ones': False, # Skip the standard value put in by analysis
            'skip-x-zeros': False,
            'skip-x-minus-ones': False, # Skip the standard value put in by analysis
            'switch-xy': False, # Switch the x and y axis values
            'style': ['line', 'dots', 'rings'],
            'type': '',
            'line-width': 2,
            'colors': [],
            'opacity':1,
            'label': '',
            'alternate-axis': False,
        }

        # Set default arguments
        for key in kwargs:
            if key in defaults:
                defaults[key] = kwargs[key]

        # If x has combined data points, split them up
        if isinstance(x[0], tuple) or isinstance(x[0], list):
            if len(x[0]) == 1:
                x = [i[0] for i in x]
            elif len(x[0]) == 2:
                x, y = zip(*x)
            else:
                x, y, err = zip(*x)

        # If there's only one value, set the other to arange
        if y == []:
            y = x
            x = [i+1 for i in range(len(y))]

        args = setargs(defaults, kwargs)
        args['stroke'] = color(args['stroke'])
        clr = color(args['color'])
        if args['switch-xy']:
            x, y = y, x

        x, y = np.array(x), np.array(y)
        x, y = self._skip_non_data(x, y, args)
        self._data.append((x, y, args['errors'], clr, args))
        return self

    def hbox(self, range, **kwargs):
        """
        Add a horizontal box
        """

        defaults = {
            'color': 'gray',
            'opacity':1,
        }

        # Set default arguments
        for key in kwargs:
            if key in defaults:
                defaults[key] = kwargs[key]

        clr = color(defaults['color'])
        opac = defaults['opacity']

        self._shaded.append(('h', range[0], range[1], clr, opac))


    def trace(self, **kwargs):
        """
        Plot a trace and keep it simple.
        """

        defaults = {
            'xunits': '',
            'yunits': '',
            'simplified-dpi': 400,
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Set the axis to be blank
        ax = self._noaxis(ax)

        # Loop through all saved data and plot
        mnx, mxx = None, None
        for x, y, err, clr, xyargs in self._data:
            if args['simplified-dpi'] > 0:
                x, y = self._simplified(x, y, args['simplified-dpi']*self._size[0])

            if mnx == None:
                mnx, mxx = x[0], x[0]

            if np.min(x) < mnx:
                mnx = np.min(x)
            if np.max(x) > mxx:
                mxx = np.max(x)

            ax.plot(x, y, linewidth=0.5, alpha=1, color=clr)
            ax.set_xlim((mnx, mxx))

        scalebar(ax, **args)

        self._end_graph(fig, ax, args)
        return self

    def temp_jetonwhite(self):
        supmap = {
            'red': (
                (0.0, 79.0/256, 79.0/256),
                (0.5, 0, 0), # Mapped with (1 - (1 - old)*0.9)
                (1.0, 1, 1), # Moved Jet start here
            ),
            'green': (
                (0.0, 209.0/256, 209.0/256),
                (0.5, 0, 0),
                (1.0, 0, 0),
            ),
            'blue': (
                (0.0, 168.0/256, 168.0/256),
                (0.5, 0, 0),
                (1.0, 0, 0), # Moved Jet start here
            )
        }

        return mpl.colors.LinearSegmentedColormap('my_colormap', supmap, 256)

    def temp_scatter(self, **kwargs):
        defaults = {}

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        ax = self._simpleaxis(ax)

        # Loop through all saved data and plot
        for x, y, err, clr, xyargs in self._data:
            #if args['simplified-dpi'] > 0:
            #	x, y = self._simplified(x, y, args['simplified-dpi']*self._size[0])

            #x, y, ignoreerr = self._order_and_combine_data(subset['data'][cell], args)

            clr = clr if 'colors' not in xyargs else xyargs['colors']
            print(len(x), len(y), len(xyargs['colors']), xyargs['colors'][0])
            ax.scatter(x, y, vmin=0.0, vmax=1.0, c=(1.0 - xyargs['colors']),
                       edgecolor='none', alpha=0.6, cmap=self.temp_jetonwhite())

        self._end_graph(fig, ax, args)
        return self

    # Line graph on axis
    def line(self, **kwargs):
        """
        Plot a line graph
        """

        defaults = {
            'combine-within': ['mean', 'median', 'max', 'average', 'none'],  # | 'median' | 'max'
            'tolerance': 0.01,
            'error-type': ['stdev', 'stderr'],
            'average-by-type': False,
            'average': False,
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'xticks': [],
            'yticks': [],

            'alt-xmin': None,
            'alt-xmax': None,
            'alt-ymin': None,
            'alt-ymax': None,
            'alt-yscale': ['linear', 'log', 'sym-log'],  # | 'log' | 'sym-log'

            'hide-y': False,
            'dots': False,

            'xscale': ['linear', 'log', 'sym-log'],  # | 'log' | 'sym-log'
            'yscale': ['linear', 'log', 'sym-log'],  # | 'log' | 'sym-log'
            'legend': False,
            'legend-title': '',

            # NOTE: plots relative error for 'stdev' in log plot
            #'simplified-dpi': 0, Add simplification in the future
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        ax = self._simpleaxis(ax)

        # Loop through all saved data and plot
        altax = False
        for x, y, err, clr, xyargs in self._data:
            #if args['simplified-dpi'] > 0:
            #	x, y = self._simplified(x, y, args['simplified-dpi']*self._size[0])

            #x, y, ignoreerr = self._order_and_combine_data(subset['data'][cell], args)

            if xyargs['alternate-axis']:
                altax = True
            else:
                if len(err) > 0:
                    ax.fill_between(x, y+err, y-err, facecolor=clr, edgecolor=None, alpha=0.2, linewidth=0.0)

                # Check if linewidth is set to 0, if so, fill between 0 and values
                if xyargs['line-width'] <= 0:
                    ax.fill_between(x, [0 for i in y], y, facecolor=clr, edgecolor=None, alpha=xyargs['opacity'])
                else:
                    ax.plot(x, y, linewidth=xyargs['line-width'], alpha=xyargs['opacity'], color=clr, label=xyargs['label'])

                if args['dots']:
                    ax.scatter(x, y, facecolor='none', marker='o', lw=2, s=70, edgecolor=clr, alpha=0.4)

        if altax:
            ax2 = ax.twinx()
            ax2 = self._simpleaxis(ax2)

            for x, y, err, clr, xyargs in self._data:
                if xyargs['alternate-axis']:
                    if len(err) > 0:
                        ax2.fill_between(x, y + err, y - err, facecolor=clr, edgecolor=None, alpha=0.2,
                                        linewidth=0.0)

                    ax2.plot(x, y, linewidth=xyargs['line-width'], alpha=xyargs['opacity'], color=clr,
                            label=xyargs['label'])
                    if args['dots']:
                        ax2.scatter(x, y, facecolor='none', marker='o', lw=2, s=70, edgecolor=clr, alpha=0.4)

            self.setaxes(ax2, args, True)

        # for subset in data:
        # 	av = []

        # 	for cell in subset['data']:
        # 		clr = self._getlabelcolor(cell, args, subset['color'] if 'color' in subset else '',
        # subset['by'] if 'by' in subset else '')
        # 		x, y, ignoreerr = self._order_and_combine_data(subset['data'][cell], args)
        # 		for xd, yd in zip(x, y): av.append((xd, yd))

        # 		if not args['error']:
        # 			if args['dots']:
        # 				ax.plot(x, y, linewidth=0.5, alpha=0.2, color=clr)
        # 				ax.scatter(x, y, facecolor='none', marker='o', lw=2, s=70, edgecolor=clr, alpha=0.4)
        # 			else: ax.plot(x, y, linewidth=0.5, alpha=0.5 if args['fit'] == 'none' else 0.2, color=clr)

        # 	if args['fit'] != 'none':
        # 		x, y, std = self._fit(av, args, subset['title'] if 'title' in subset else '')
        # 		if args['error']:
        # 			bot, top = self._error(x, y, std, args)
        # 			ax.fill_between(x, bot, top, color=clr, alpha=0.3)

        # 		ax.plot(x, y, linewidth=5, alpha=1, color=clr, solid_capstyle='round')
        # 		#if not args['dots']: ax.plot(x, y, linewidth=5, alpha=0.2, color=self.color('black'), solid_capstyle='round')

        if args['legend']:
            legend = plt.legend(fontsize=12, title=args['legend-title'])
            plt.setp(legend.get_title(), fontsize=12)
        self._end_graph(fig, ax, args)
        return self

    def axes(self, **kwargs):
        defaults = {}
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()
        return fig, ax, args

    def scatter(self, **kwargs):
        """
        Plot a scatter plot
        """

        defaults = {
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'hide-y': False,
            'clip-on': True,
            'line': ([], []),
            'line-color': 'gray',
            'best-fits': False,
            'best-fit': False,
            'fit-type': 'glm',  # {'linear', 'GLM'}
            'fit-type-family': 'Gaussian',  # statsmodels.api.families.FAMILY
            'fit-type-link': 'identity',  # statsmodels.api.families.links.LINK
            'fit-error': True,
            'tiny': False,

            'xscale': ['linear', 'log', 'symlog'],  # | 'log' | 'sym-log'
            'yscale': ['linear', 'log', 'symlog'],  # | 'log' | 'sym-log'

            'legend': False,
            'legend-title': '',
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        ax = self._simpleaxis(ax)

        fx = []
        fy = []
        # Loop through all saved data and plot
        for x, y, err, clr, xyargs in self._data:
            clr = clr if 'colors' not in xyargs or len(xyargs['colors']) == 0 \
                else xyargs['colors']
            clr2 = '#CCCCCC' if len(xyargs['color2']) == 0 else xyargs['color2']

            if err is not None and len(err) == len(y):
                for ex, ey, er in zip(x, y, err):
                    ax.plot(
                        [ex, ex], [ey+er, ey-er], linewidth=1, color=clr,
                        alpha=0.5)

            if args['tiny']:
                ax.scatter(
                    x, y, edgecolor=None, alpha=0.05, color=clr, s=10,
                    clip_on=args['clip-on'], label=xyargs['label'])
            elif xyargs['style'] != 'rings':
                ax.scatter(
                    x, y, edgecolor=clr2, alpha=xyargs['opacity'],
                    color=clr, s=70, clip_on=args['clip-on'], label=xyargs['label'])
            else:
                ax.scatter(
                    x, y, edgecolor=clr, alpha=xyargs['opacity'],
                    color='none', s=70, clip_on=args['clip-on'], label=xyargs['label'])
            if args['best-fits']:
                if 'log' not in args['yscale'] and \
                        args['fit-type'][:3] == 'lin':
                    ax.plot(
                        np.unique(x),
                        np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
                        lw=2, color=clr)
                # elif args['fit-type'][:3] == 'log':
                #     fitx, fity = fitdecay(x, y)
                #     ax.plot(fitx, fity, lw=2, color=clr)
                # elif 'log' in args['yscale']:
                #     logy = np.poly1d(
                #         np.polyfit(x, np.log10(y), 1))(np.unique(x))
                #     ax.plot(np.unique(x), np.power(10, logy), lw=2, color=clr)
                elif args['fit-type'].lower() == 'glm':
                    fitx, fity, terr, berr = fit_GLM(
                        x, y, family=args['fit-type-family'],
                        link=args['fit-type-link'])

                    if args['fit-error']:
                        ax.fill_between(fitx, terr, berr, facecolor=clr, edgecolor=None, alpha=0.2, linewidth=0.0)
                    ax.plot(fitx, fity, lw=2, color=clr)

            fx = np.concatenate([fx, x])
            fy = np.concatenate([fy, y])

        if args['best-fit']:
            if 'log' not in args['yscale'] and args['fit-type'][:3] == 'lin':
                ax.plot(
                    np.unique(fx),
                    np.poly1d(np.polyfit(fx, fy, 1))(np.unique(fx)),
                    lw=2, color=args['line-color'])
            elif 'log' in args['yscale']:
                logy = np.poly1d(
                    np.polyfit(fx, np.log10(fy), 1))(np.unique(fx))
                ax.plot(
                    np.unique(fx), np.power(10, logy), lw=2,
                    color=args['line-color'])
            elif args['fit-type'][:3] == 'log':
                fitx, fity = fitdecay(fx, fy)
                ax.plot(fitx, fity, lw=2, color=clr)
            elif args['fit-type'].lower() == 'glm':
                fitx, fity, terr, berr = fit_GLM(
                    fx, fy, family=args['fit-type-family'],
                    link=args['fit-type-link'])
                ax.fill_between(fitx, terr, berr, facecolor=clr, edgecolor=None, alpha=0.2, linewidth=0.0)
                ax.plot(fitx, fity, lw=2, color=clr)

        elif len(args['line'][0]) > 1 and \
                len(args['line'][0]) == len(args['line'][1]):
            ax.plot(
                args['line'][0], args['line'][1],
                color=args['line-color'], lw=2, clip_on=args['clip-on'])

        # Save or plot the data
        if args['legend']:
            legend = plt.legend(fontsize=12, title=args['legend-title'])
            plt.setp(legend.get_title(), fontsize=12)

        self._end_graph(fig, ax, args)
        return self

    def bar(self, **kwargs):
        """
        Plot a scatter plot
        """

        defaults = {
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'hide-y': False,
            'clip-on': True,
            'xlabels': [],

            'xscale': ['linear', 'log', 'sym-log'],  # | 'log' | 'sym-log'
            'yscale': ['linear', 'log', 'sym-log'],  # | 'log' | 'sym-log'

            'legend': False,
            'legend-title': '',
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        ax = self._simpleaxis(ax)

        fx = []
        fy = []
        # Loop through all saved data and plot

        barwidth = 0.8/len(self._data)
        off = 0

        xlabelinds = []
        for x, y, err, clr, xyargs in self._data:
            clr = clr if 'colors' not in xyargs or len(xyargs['colors']) == 0 else xyargs['colors']
            if len(xlabelinds) == 0: xlabelinds = x

            if len(xyargs['color2']) > 0:
                mpl.rcParams['hatch.linewidth'] = 4.0
                ax.bar(x + off, y, barwidth, color=clr, yerr=None if len(xyargs['errors']) == 0 else xyargs[
                    'errors'], edgecolor=xyargs['color2'], hatch='//', label=xyargs['label'])
            else:
                tbar = ax.bar(x + off, y, barwidth, color=clr, yerr=None if len(xyargs['errors']) == 0 else xyargs[
                        'errors'], label=xyargs['label'])

            off += barwidth

        if len(args['xlabels']) > 0:
            ax.set_xticks(xlabelinds)  # + width/2)
            ax.set_xticklabels(args['xlabels'])

        # Save or plot the data
        if args['legend']:
            legend = plt.legend(fontsize=12, title=args['legend-title'])
            plt.setp(legend.get_title(), fontsize=12)
        self._end_graph(fig, ax, args, True)

        return self

    def polygons(self, **kwargs):
        """
        Plot a series of polygons
        :param kwargs:
        :return:
        """

        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        defaults = {
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None,
            'hide-y': False,
            'clip-on': True,
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        ax = self._simpleaxis(ax)

        # Loop through all saved data and plot
        patches = []
        for x, y, err, clr, xyargs in self._data:
            clr = clr if 'colors' not in xyargs or len(xyargs['colors']) == 0 else xyargs['colors']

            pxy = np.zeros((len(x), 2))
            pxy[:, 0] = np.array(x)
            pxy[:, 1] = np.array(y)

            poly = Polygon(pxy, True, lw=3, color=clr, facecolor=clr, fill=True,
                           alpha=xyargs['opacity'], ec=xyargs['stroke'], clip_on=args['clip-on'])
            ax.add_patch(poly)
            # patches.append(poly)

        # pc = PatchCollection(patches)
        # ax.add_collection(pc)

        # Save or plot the data
        self._end_graph(fig, ax, args)
        return self

    def axis(self, **kwargs):
        """
        Return a graph axis, formatted correctly.
        """

        defaults = {
            'simple-axis': True,
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)
        fig, ax = self._init_graph()

        # Simplify the axis, keynote style
        if args['simple-axis']:
            ax = self._simpleaxis(ax)

        self._fig = fig

        return ax

    def graph(self, axis, **kwargs):
        """
        Return a graph axis, formatted correctly.
        """

        defaults = {
        }

        # Generate arguments by combining general graph defaults,
        # specific graph defaults, and kwargs
        args = setargs(dict(self.graph_defaults, **defaults), kwargs)

        self._end_graph(self._fig, axis, args)

    def colorbar(self, pcolormesh):
        self._fig.colorbar(pcolormesh)


    graph_defaults = {
        'title': '',
        'xtitle': '',
        'ytitle': '',
        'font': ['Gotham', 'Helvetica Neue', 'Helvetica', 'Arial'],
        'save': '', # non-zero len str is save path
        'format': ['pdf', 'png'], # | 'png'
    }


    defaults = {
        'combine': ['mean', 'median', 'max', 'average', 'none'], # | 'median' | 'max'
        'tolerance': 0.01,
        'error-type': ['stdev', 'stderr'], # | 'stderr' NOTE: plots relative error for 'stdev' in log plot
        'skip-zeros': False, # Skip zero values for averaging and plotting
        'skip-minus-ones': True, # Skip the standard value put in by analysis
        'skip-x-zeros': False,
        'skip-x-minus-ones': True, # Skip the standard value put in by analysis
        'switch-xy': False, # Switch the x and y axis values
        'color': 'gray',
    }


    default = {
        'title': '',
        'xtitle': '',
        'ytitle': '',
        'column-names': False,
        'title-color': 'black',
        'font': ['Gotham', 'Helvetica Neue', 'Helvetica', 'Arial'],
        'xscale': ['linear', 'log', 'sym-log'], # | 'log' | 'sym-log'
        'yscale': ['linear', 'log', 'sym-log'], # | 'log' | 'sym-log'
        'hide-y': False, # hides y axis
        'half-width': False, # sets width to half of a slide

        'labels':{}, # List of labels of individual cells

        'save': '', # non-zero len str is save path
        'format': ['pdf', 'png'], # | 'png'
    }

    # =========================================================================================== #
    # Internal functions

    def _simplified(self, x, y, width=0):
        """
        Simplify a trace so that it is visually correct, but is possible
        to include in a figure for a paper. X is optional.
        """

        # Take care of optional arguments
        if isinstance(y, int) or isinstance(y, float):
            width = y
            y = x
            x = []

        # Take care of xs if necessary
        if len(x) == 0:
            x = np.arange(len(y))

        # Set the step size and return if already simple enough
        step = int(math.floor(len(y)/width))
        if step < 2:
            return (np.array(x), np.array(y))

        # Prepare for indexing
        x = np.array(x)
        y = np.array(y)

        newx = []
        newy = []
        for i in range(0, len(y), step):
            sub = y[i:i+step]
            mni = np.argmin(sub)
            mxi = np.argmax(sub)
            if mni < mxi:
                newx.extend([x[i+mni], x[i+mxi]])
                newy.extend([sub[mni], sub[mxi]])
            else:
                newx.extend([x[i+mxi], x[i+mni]])
                newy.extend([sub[mxi], sub[mni]])

        newx.append(x[-1])
        newy.append(y[-1])
        return (np.array(newx), np.array(newy))

    def _skip_non_data(self, x, y, args):
        """
        Skip data coded with 0s or -1s.
        """

        # Check if skipping is necessary
        modify = False
        for key in args:
            if key[:4] == 'skip':
                modify += args[key]

        if not modify:
            return x, y
        else:
            # Run through only once, given that often one skips multiple types
            xx, yy = [], []
            for i, j in zip(x, y):
                if args['skip-zeros'] and (i == 0 or j == 0):
                    pass
                elif args['skip-minus-ones'] and (i == -1 or j == -1):
                    pass
                elif args['skip-x-zeros'] and i == 0:
                    pass
                elif args['skip-x-minus-ones'] and i == -1:
                    pass
                else:
                    xx.append(i)
                    yy.append(j)
            return xx, yy

    def _init_graph(self):
        self._setfont()
        self._setlines()

        # Set the width of the figure to be half-width if the half-width flag is true
        # Initialize the figure and axis
        fig = plt.figure(figsize=self._size)
        ax = plt.subplot(111)

        # Set axis within graph type
        return fig, ax

    def _end_graph(self, fig, ax, args={}, skipx=False):
        # Fix the axes, dependent on graph type
        # Must be done after the plotting
        # Then set the values of the titles
        self._drawboxes(ax)
        self.setaxes(ax, args, skipx=skipx)
        self.settitles(ax, args)

        # Save the graph if desired, otherwise show the graph
        with SuppressErrors():
            if len(args['save']) > 0:
                filename = ''.join([args['save'], '.', args['format']])
                plt.savefig(opath.join(self._directory, filename), transparent=True)
            else:
                plt.show()
        plt.close(fig)

    def _drawboxes(self, ax):
        for b in self._shaded:
            if b[0] == 'h':
                ax.axvspan(b[1], b[2], color=b[3], alpha=b[4], lw=0)
            else:
                ax.axhspan(b[1], b[2], color=b[3], alpha=b[4], lw=0)
        return self

    def _noaxis(self, ax):
        """
        Set axis to just a scale bar.
        """
        # Ref: https://gist.github.com/dmeliza/3251476

        ax.axis('off')
        return ax

    # Set axis to look like my axes from Keynote into Illustrator
    def _simpleaxis(self, ax):
        ax.yaxis.grid(True, linestyle='solid', linewidth=0.75, color='#AAAAAA')
        ax.tick_params(axis='x', pad = 15)
        ax.tick_params(axis='y', pad = 15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_xaxis().set_tick_params(direction='out')
        ax.get_yaxis().set_ticks_position('none')
        ax.set_axisbelow(True)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        return ax

    def _setfont(self, font="Gotham"):
        """
        Set font for numerals. Suggested alternatives to Gotham are
        Montserrat, Helvetica Neue, Helvetica, and Arial.
        """

        import matplotlib as mpl
        f = {'family':font, 'weight':'light', 'size':20}
        mpl.rc('font', **f)
        return self

    def _setlines(self):
        """
        Set the line widths to be subtle, like my Keynote graphs
        """

        import matplotlib as mpl
        mpl.rcParams['lines.linewidth'] = 0.75
        return self

    # Order X values from low to high and combine Y values with equal Xs or Xs within tolerance
    def _order_and_combine_data(self, data, args):
        """

        """
        data.sort()
        i = 0
        xl = []
        yl = []
        err = []
        while i < len(data):
            if (not args['skip-x-minus-ones'] or abs(data[i][0] + 1.0) > args['tolerance']) and (
                    not args['skip-x-zeros'] or abs(data[i][0]) > args['tolerance']):
                x = data[i][0]
                y = [data[i][1]]
                n = 1
                while i + n < len(data) and (isinstance(data[i + n][0], str) or data[i + n][0] - x <= args['tolerance']):
                    if not isinstance(data[i + n][0], str):
                        if not args['skip-zeros'] or data[i+n][1] != 0:
                            if not args['skip-minus-ones'] or abs(data[i+n][1] + 1.0) > args['tolerance']:
                                y.append(data[i+n][1])
                    n += 1

                oy = self.combine(y, args)
                if not args['skip-zeros'] or oy != 0:
                    if not args['skip-minus-ones'] or abs(oy + 1.0) > args['tolerance']:
                        xl.append(x)
                        yl.append(oy)
                        err.append(np.std(y))
                        err[-1] = err[-1]/len(y) if args['error-type'] == 'stderr' else err[-1]

                i += n
            else: i += 1

        return (np.array(xl), np.array(yl), np.array(err))

    # Hide y if necessary
    def setaxes(self, ax, args, alt=False, skipx=False):
        a = 'alt-' if alt else ''

        if a+'hide-y' in args and args[a+'hide-y']:
            ax.get_yaxis().set_ticklabels([])
            ax.set_ylabel(' ')

        if not skipx:
            if a+'xscale' in args: ax.set_xscale(args[a+'xscale'])
            if a + 'xmin' in args and args[a + 'xmin'] != None: ax.set_xlim([args[a + 'xmin'], ax.get_xlim()[1]])
            if a + 'xmax' in args and args[a + 'xmax'] != None: ax.set_xlim([ax.get_xlim()[0], args[a + 'xmax']])
            if a + 'hide-x-ticks' in args and args[a + 'hide-x-ticks']: ax.get_xaxis().set_ticklabels([])
            if a+'xticks' in args and len(args[a+'xticks']) > 0: ax.set_xticks(args[a+'xticks'])

        if a+'yscale' in args:
            if args[a+'yscale'].find('log') > -1: ax.set_yscale(args[a+'yscale'], nonposy='clip')
            else: ax.set_yscale(args[a+'yscale'])

        if a+'ymin' in args and args[a+'ymin'] != None: ax.set_ylim([args[a+'ymin'], ax.get_ylim()[1]])
        if a+'ymax' in args and args[a+'ymax'] != None: ax.set_ylim([ax.get_ylim()[0], args[a+'ymax']])

        if a+'yticks' in args and len(args[a+'yticks']) > 0: ax.set_yticks(args[a+'yticks'])

    # Set the X, Y, and supertitles
    def settitles(self, ax, args):
        import matplotlib.pyplot as plt

        if 'xtitle' in args and len(args['xtitle']) > 0:
            ax.set_xlabel(args['xtitle'])
            plt.subplots_adjust(bottom=0.20)

        if 'ytitle' in args and len(args['ytitle']) > 0:
            ax.set_ylabel(args['ytitle'])
            if 'half-width' in args:
                if args['half-width']: plt.subplots_adjust(left=0.20)
            elif self._size[0] < 10: plt.subplots_adjust(left=0.20)

        if 'title' in args and len(args['title']) > 0:
            #correctedtitle = args['title'].replace("\\n", "\n") if args['title'].find("\n") > -1 else "\n" + args['title']
            correctedtitle = args['title'].replace("|", "\n") if args['title'].find("|") > -1 else "\n" + args['title']
            if 'save' in args and args['save'] != '':
                plt.suptitle(correctedtitle, **{'family':str(args['font']), 'weight':'book',
                                                'size':24, 'va':'top', 'y':0.995, 'color':'black'})
            else:
                # Due to an error in the MacOSX backend
                plt.suptitle(correctedtitle, **{'size':24, 'va':'top', 'y':0.995, 'color':'black'})
            plt.subplots_adjust(top=0.84)


def graph(save_directory='', width='full'):
    """
    Make a grapher instance and return it. Optionally take save
    directory and width, which can be full, half, or an integer.
    """

    out = Grapher(save_directory, width)
    return out
