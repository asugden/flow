from re import match as rematch
import matplotlib.colors as mplcolors

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
        'blue': '#47AEED',  # previously 4087DD
        'yellow': '#F2E205',
    }

    if isinstance(name, int):
        clrs = [colors[key] for key in colors]
        return clrs[name%len(clrs)]
    elif name in colors:
        return colors[name]
    elif rematch('^(#[0-9A-Fa-f]{6})|(rgb(a){0,1}\([0-9,]+\))$', name):
        return name
    else:
        return colors['gray']


def hex2tuple(clr):
    """
    Convert a color in hex to a tuple. Can add the ability to handle
    rgb(a) strings in the future.
    """
    # Check if already tuple, otherwise return as string
    if isinstance(clr, tuple):
        return clr
    elif clr[0] == '#' and len(clr) == 7:
        return tuple(int(clr[i:i + 2], 16) for i in (1, 3, 5))
    else:
        print 'WARNING: Color format not recognized.'


def tuple2hex(clr):
    """
    Convert a color in hex to a tuple. Can add the ability to handle
    rgb(a) strings in the future.
    """
    # Check if already tuple, otherwise return as string
    if len(clr) != 3:
        return clr
    else:
        return '#%02x%02x%02x'%(int(clr[0]), int(clr[1]), int(clr[2]))

def numpy2hex(clr):
    """
    Convert a vector of 3-4 floats [0-1] into a hex value
    :param clr: vector of 3-4 floats[0-1]
    :return: hex string
    """

    if len(clr) < 3: return clr
    clr = [int(round(255*c)) for c in clr[:3]]
    return tuple2hex(clr)

def numpy2rgba(clr):
    """
    Convert a color with 3-4 floats of values 0-1 to rgba string.
    :param clr: vector of length 3-4 with all values 0-1
    :return: rgba string representation
    """

    if len(clr) < 3 or len(clr) > 4: return clr
    if len(clr) == 3: clr = list(color) + [1]
    for i in range(3): clr[i] = int(round(255*clr[i]))
    return 'rgba(%i,%i,%i,%i)' % tuple(clr)

def numpy2rgb(clr):
    """
    Convert a color with 3-4 floats of values 0-1 to rgba string.
    :param clr: vector of length 3-4 with all values 0-1
    :return: rgba string representation
    """

    if len(clr) < 3: return clr
    for i in range(3): clr[i] = int(round(255*clr[i]))
    return 'rgb(%i,%i,%i)' % tuple(clr[:3])


def merge(clr, val, mergeto=(255, 255, 255)):
    """
    Merge colors (defined in a list clrs) together with the remaining
    fraction of the color being mergeto (vals should sum to < 1). Colors
    should be tuples.
    :param clr uint8 rgb list
    """

    # Return mergeto if vals and clrs are empty
    if len(clr) == 0: return mergeto
    clr = hex2tuple(clr)
    mergeto = hex2tuple(mergeto)

    # Find out how much of value comes from mergeto
    # if not isinstance(vals, list): vals = [vals, vals, vals]
    out = [0, 0, 0]
    mergetoval = 1.0 - val

    # Iterate over R, G, and B
    for i in range(3):
        out[i] += val*clr[i]
        out[i] += mergetoval*mergeto[i]

    out = [int(round(i)) for i in out]
    return out

def gradbr():
    # Makes a color map with the span of jet PLUS black
    supmap = {
        'red': (
            (0.0, 71./255, 71./255),
            (0.5, 1, 1),
            (1.0, 214./255, 214./255),
        ),
        'green': (
            (0.0, 174./255, 174./255),
            (0.5, 1, 1),
            (1.0, 30./255, 30./255),
        ),
        'blue': (
            (0.0, 237./255, 237./255),
            (0.5, 1, 1),
            (1.0, 33./255, 33./255),
        )
    }

    return mplcolors.LinearSegmentedColormap('my_purple_colormap', supmap, 256)
