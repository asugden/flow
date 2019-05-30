import math
import numpy as np

from . import svggraph


def assign_cells_from_date(date, path, colors,
                           inner_spacing=20,
                           outer_spacing=10,
                           inner_threshold=0.1,
                           inner_connections=True,
                           outer_connections=False,
                           inner_dots_as_pies=False,
                           outer_dots_as_pies=True,
                           min_to_display=0.01,
                           dot_scale_factor=1):
    """
    Make a circle cell plot from a date.

    Parameters
    ----------
    date : Date instance
    path : str
        Save path
    colors : dict
        key: stimulus name, val: color code
    inner_spacing : float
        Space between dots in inner ring
    outer_spacing : float
        Space between dots in outer ring
    inner_threshold : float
        Marginal threshold for being in the inner ring
    inner_connections : bool
        If true, draw arcs for connections/conditionals
    outer_connections : bool
        If true, draw arcs for connections/conditionals in outer ring
        Note, overlaps arcs in inner ring.
    inner_dots_as_pies : bool
        Rather than plotting by the max activity, plot a pie chart of activity
    outer_dots_as_pies : bool
        Rather than plotting by the max activity, plot a pie chart of activity
    min_to_display : float
        Minimum activity level to include
    dot_scale_factor : float
        Scale the dot sizes by this factor.

    """

    runs = date.runs('training')
    c2p = runs[0].classify2p()
    model = c2p.train()

    plot = PlotCircle(path, model['conditional'], colors, sizes=model['marginal'],
                      inner_spacing=inner_spacing,
                      outer_spacing=outer_spacing,
                      inner_threshold=inner_threshold,
                      inner_connections=inner_connections,
                      outer_connections=outer_connections,
                      inner_dots_as_pies=inner_dots_as_pies,
                      outer_dots_as_pies=outer_dots_as_pies,
                      min_to_display=min_to_display,
                      dot_scale_factor=dot_scale_factor)




class PlotCircle():
    """
    Make a circle plot, usually of cells, with connections between.
    """

    def __init__(self, path, connections, colors, sizes=None,
                 inner_spacing=20,
                 outer_spacing=10,
                 inner_threshold=0.1,
                 inner_connections=True,
                 outer_connections=False,
                 inner_dots_as_pies=False,
                 outer_dots_as_pies=True,
                 min_to_display=0.01,
                 dot_scale_factor=1):

        if sizes == None:

        inner, outer = _assign_cells_to_ring(classifier['marginal'], pars['inner-threshold'])
        innerrad = (len(inner)*pars['inner-spacing'])/(2*math.pi)
        outerrad = max((len(outer)*pars['outer-spacing'])/(2*math.pi), innerrad + pars['inter-spacing'])




def cellcircle(path, classifier, apars, stimulus=''):
    pars = {
        'inner-spacing': 20,
        'outer-spacing': 10,
        'inter-spacing': 20,
        'inner-threshold': 0.1,
        'inner-joints': True,
        'outer-joints': False,
        'piecharts': False,
        'outer-piecharts': True,
        'min-to-display': 0.01,
        'cell-size-scale-factor': 1,
    }
    for key in apars:
        pars[key] = apars[key]

    if stimulus == '':
        stimuli = [key for key in classifier['marginal']]
    else:
        stimuli = [stimulus]

    inner, outer = _assign_cells_to_ring(classifier['marginal'], pars['inner-threshold'])
    innerrad = (len(inner)*pars['inner-spacing'])/(2*math.pi)
    outerrad = max((len(outer)*pars['outer-spacing'])/(2*math.pi), innerrad + pars['inter-spacing'])

    # Initialize the graph with enough padding
    sz = 2*outerrad + 4*pars['outer-spacing']
    gr = svggraph.svg((sz, sz))

    innerpos = [_cell_position(i, len(inner), innerrad, (sz/2.0, sz/2.0))
                for i in range(len(inner))]
    outerpos = [_cell_position(i, len(outer), outerrad, (sz/2.0, sz/2.0))
                for i in range(len(outer))]

    if pars['inner-joints']:
        for i, c in enumerate(inner):
            probs = _cell_joint_probabilities(c, inner, classifier['conditional'], classifier['marginal'],
                                              pars['min-to-display'])
            probs = {key: np.array(probs[key])*pars['cell-size-scale-factor'] for key in probs.keys()}

            for key in probs:
                if key not in stimuli:
                    for j in range(len(probs[key])):
                        probs[key][j] = 0
            x, y = _cell_position(i, len(inner), innerrad, (sz/2.0, sz/2.0))
            _add_joint_probabilities(gr, probs, innerpos[i], innerpos, (sz/2.0, sz/2.0), innerrad*2)

    if pars['outer-joints']:
        for i, c in enumerate(outer):
            probs = _cell_joint_probabilities(c, inner, classifier['conditional'], classifier['marginal'],
                                              pars['min-to-display'])
            probs = {key: np.array(probs[key])*pars['cell-size-scale-factor'] for key in probs.keys()}

            for key in probs:
                if key not in stimuli:
                    for j in range(len(probs[key])):
                        probs[key][j] = 0
            x, y = _cell_position(i, len(outer), outerrad, (sz/2.0, sz/2.0))
            _add_joint_probabilities_oi(gr, probs, outerpos[i], innerpos, (sz/2.0, sz/2.0), outerrad*2)

    # Draw the circles
    gr.ring((sz/2.0, sz/2.0), innerrad, {'stroke': '#CCCCCC', 'opacity': 0.2})
    gr.ring((sz/2.0, sz/2.0), outerrad, {'stroke': '#CCCCCC', 'opacity': 0.2})

    # Draw the cell positions
    for i in range(len(inner)):
        x, y = _cell_position(i, len(inner), innerrad, (sz/2.0, sz/2.0))
        gr.circle((x, y), 1, {'color': '#CCCCCC'})

    for i in range(len(outer)):
        x, y = _cell_position(i, len(outer), outerrad, (sz/2.0, sz/2.0))
        gr.circle((x, y), 1, {'color': '#CCCCCC'})

    if pars['piecharts']:
        for i, c in enumerate(inner):
            probs = _cell_single_probabilities(c, classifier['marginal'], pars['min-to-display'])
            probs = {key:probs[key]*pars['cell-size-scale-factor'] for key in probs.keys()}
            x, y = _cell_position(i, len(inner), innerrad, (sz/2.0, sz/2.0))
            _add_single_probability(gr, probs, (x, y), pars['inner-spacing'])
    else:
        for i, c in enumerate(inner):
            probs = _cell_single_probabilities(c, classifier['marginal'], pars['min-to-display'])
            probs = {key:probs[key]*pars['cell-size-scale-factor'] for key in probs.keys()}
            x, y = _cell_position(i, len(inner), innerrad, (sz/2.0, sz/2.0))
            _add_greatest_probability(gr, probs, (x, y), pars['inner-spacing'], stimulus)

    if pars['outer-piecharts']:
        for i, c in enumerate(outer):
            probs = _cell_single_probabilities(c, classifier['marginal'], pars['min-to-display'])
            probs = {key:probs[key]*pars['cell-size-scale-factor'] for key in probs.keys()}
            x, y = _cell_position(i, len(outer), outerrad, (sz/2.0, sz/2.0))
            _add_single_probability(gr, probs, (x, y), pars['outer-spacing'])

    # Save
    print path
    gr.graph(path)


def _add_joint_probabilities(gr, probs, pos, allpos, center=(-1, -1), diam=1):
    """
    Add lines connecting cell at pos to all cells with strong enough
    joint probabilities.
    """

    # Iterate through stimuli
    for i, stim in enumerate(probs):
        for i in range(len(probs[stim])):
            if probs[stim][i] > 0:
                lw = probs[stim][i]*6
                op = probs[stim][i]
                # gr.line(pos, allpos[i], {'stroke':clrs[stim], 'stroke-width':lw, 'opacity':op})
                avpos = ((pos[0] + allpos[i][0])/2.0, (pos[1] + allpos[i][1])/2.0)
                d = math.sqrt((pos[0] - allpos[i][0])*(pos[0] - allpos[i][0]) +
                              (pos[1] - allpos[i][1])*(pos[1] - allpos[i][1]))
                # avcntl = ((avpos[0] + center[0])/2.0, (avpos[1] + center[1])/2.0)
                dmax = d/diam
                avcntl = (avpos[0]*(1.0 - dmax) + center[0]*dmax,
                          avpos[1]*(1.0 - dmax) + center[1]*dmax)

                gr.quadcurve(pos, allpos[i], avcntl, {'stroke': settings.colors(stim), 'stroke-width': lw, 'opacity':
                    op})


def _add_joint_probabilities_oi(gr, probs, pos, allpos, center=(-1, -1), diam=1):
    """
    Add lines connecting cell at pos to all cells with strong enough
    joint probabilities.
    """

    # Iterate through stimuli
    for i, stim in enumerate(probs):
        for i in range(len(probs[stim])):
            if probs[stim][i] > 0:
                lw = probs[stim][i]*6
                op = probs[stim][i]
                gr.line(pos, allpos[i], {'stroke': settings.colors(stim), 'stroke-width': lw, 'opacity': op})
            # avpos = ((pos[0] + allpos[i][0])/2.0, (pos[1] + allpos[i][1])/2.0)
            # d = math.sqrt((pos[0] - allpos[i][0])*(pos[0] - allpos[i][0]) +
            # 	(pos[1] - allpos[i][1])*(pos[1] - allpos[i][1]))
            # #avcntl = ((avpos[0] + center[0])/2.0, (avpos[1] + center[1])/2.0)
            # dmax = d/diam
            # avcntl = (avpos[0]*(1.0 - dmax) + center[0]*dmax,
            # 	avpos[1]*(1.0 - dmax) + center[1]*dmax)

            # gr.quadcurve(pos, allpos[i], avcntl, {'stroke':clrs[stim], 'stroke-width':lw, 'opacity':op})


def _cell_single_probabilities(c, probs, minprob):
    """
    Return a dict of cell firing probabilities across stimuli.
    """

    out = {}
    for key in probs:
        if probs[key][c, 0] < minprob:
            out[key] = 0
        else:
            out[key] = probs[key][c, 0]
    return out


def _cell_joint_probabilities(c, cells, probs, probsingle, minprob):
    """
    Return a dict of joint cell firing with all cells of the inner ring.
    """

    out = {}
    for key in probs:
        out[key] = []
        for jc in cells:
            if probsingle[key][c, 0]*probs[key][c, jc, 0] < minprob:
                out[key].append(0)
            elif jc == c:  # or probs[key][c, jc, 0]*probs[key][c, jc, 0] < minprob:
                out[key].append(0)
            else:
                out[key].append(probsingle[key][c, 0]*probs[key][c, jc, 0])
    return out

def _add_single_probability(gr, probs, pos, rad):
    """
    Make a chart that looks like a pie chart based on cell firing
    probability and specificity.
    """

    # Get the total firing for specificity measure
    total = 0
    nactive = 0
    for key in probs:
        total += probs[key]
        if probs[key] > 0:
            nactive += 1

    # Iterate through stimuli
    activestim = 0
    for i, stim in enumerate(probs):
        if probs[stim] > 0:
            r = probs[stim]*rad
            clr = gr.colorscale2(probs[stim]/total, settings.colors(stim))
            theta = 360.0/nactive
            start = -90 + theta*activestim

            if nactive == 1:
                gr.circle(pos, r, {'color': clr, 'stroke': settings.colors(stim), 'stroke-width': 1})
            else:
                gr.arc(pos, r, (start, start + theta), {'color': clr, 'stroke': settings.colors(stim),
                                                        'stroke-width': 1})
                activestim += 1

def _add_greatest_probability(gr, probs, pos, rad, stim=''):
    """
    Make a chart that looks like a pie chart based on cell firing
    probability and specificity.
    """

    # Get the total firing for specificity measure
    total = 0
    mxactive = ''
    mxactivity = 0
    for key in probs:
        total += probs[key]
        if probs[key] > mxactivity:
            mxactivity = probs[key]
            mxactive = key

    # Iterate through stimuli
    stim = stim if len(stim) > 0 else mxactive
    if probs[stim] > 0:
        r = probs[stim]*rad
        clr = gr.colorscale2(probs[stim]/total, settings.colors(stim))

        gr.circle(pos, r, {'color': clr, 'stroke': settings.colors(stim), 'stroke-width': 1})

def _cell_position(cell, n, rad, center):
    """
    Define the cell position using simple geometry.
    """

    theta = 360.0/n
    angle_from_top = math.radians(-90 + theta*cell)

    x = rad*math.cos(angle_from_top) + center[0]
    y = rad*math.sin(angle_from_top) + center[1]

    return x, y


def _assign_cells_to_ring(singleprobs, threshold):
    """
    Assign cells to the inner or outer ring based on their probability
    of firing during the stimulus with a threshold threshold.
    """

    inner = []
    outer = []
    probkeys = [key for key in singleprobs]

    # Iterate over all of the cells
    for i in range(np.shape(singleprobs[probkeys[0]])[0]):
        # Iterate over each condition to find where it fires the most
        maxfiring = 0
        for condition in probkeys:
            if singleprobs[condition][i, 0] > maxfiring:
                maxfiring = singleprobs[condition][i, 0]

        # Add to the appropriate list
        if maxfiring >= threshold:
            inner.append(i)
        else:
            outer.append(i)

    return inner, outer