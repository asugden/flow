# badly written, but can easily be moved, required by cellcircle.py

import math
import numpy as np
from grapher import color
import cairosvg

class SVGraph:
    def __init__(self, size=(400, 300)):
        """Initialize the SVG file with a size in pixels."""

        self.size = size
        self.t = """<?xml version="1.0" encoding="utf-8"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"><svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="%ipx" height="%ipx" viewBox="0 0 %i %i" enable-background="new 0 0 %i %i" xml:space="preserve">""" % (size[0], size[1], size[0], size[1], size[0], size[1])

    def square(self, pos, sz=(4, 4), pars={}):
        default = {
            'color': color('gray'),
            'stroke': 'none',
            'stroke-width': 0,
        }

        for p in pars: default[p] = pars[p]
        self.t += '<rect x="%i" y="%i" width="%i" height="%i" style="fill:%s; stroke-width: %i; stroke: %s"/>' % (
            pos[0], pos[1], sz[0], sz[1], default['color'], default['stroke-width'], default['stroke'])

    def line(self, p1, p2, pars={}):
        default = {
            'stroke': color('gray'),
            'stroke-width': 1,
            'opacity': 1.0,
        }

        for p in pars: default[p] = pars[p]
        self.t += '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" style="stroke-width: %.2f; stroke: %s; opacity: %f;"/>' % (
            p1[0], p1[1], p2[0], p2[1], default['stroke-width'], self._hex(default['stroke']), default['opacity'])

    def quadcurve(self, p1, p2, control, pars={}):
        default = {
            'stroke': color('gray'),
            'stroke-width': 1,
            'opacity': 1.0,
        }

        for p in pars: default[p] = pars[p]
        self.t += '<path d="M %.2f %.2f Q %.2f %.2f %.2f %.2f" style="fill:none; stroke-width: %.2f; stroke: %s; opacity: %f;"/>' % (
            p1[0], p1[1], control[0], control[1], p2[0], p2[1], default['stroke-width'], self._hex(default['stroke']), default['opacity'])

    def circle(self, pos, rad, pars={}):
        default = {
            'color': color('gray'),
            'stroke': 'none',
            'stroke-width': 0,
            'opacity': 1.0,
        }

        for p in pars: default[p] = pars[p]
        self.t += '<circle cx="%.2f" cy="%.2f" r="%.2f" style="fill:%s; stroke-width: %i; stroke: %s; opacity: %f;"/>' % (
            pos[0], pos[1], rad, self._hex(default['color']), default['stroke-width'],
            self._hex(default['stroke']), default['opacity'])

    def ring(self, pos, rad, pars={}):
        default = {
            'stroke': color('gray'),
            'stroke-width': 2,
            'opacity': 1.0,
        }

        for p in pars: default[p] = pars[p]
        self.t += '<circle cx="%.2f" cy="%.2f" r="%.2f" style="fill: none; stroke-width: %i; stroke: %s; opacity: %f;"/>' % (
            pos[0], pos[1], rad, default['stroke-width'],
            self._hex(default['stroke']), default['opacity'])

    def arc(self, pos, r, a, pars={}):
        """
        Draw an svg arc centered on tuple pos, with radius r, with an
        arc beginning at a[0] and going to a[1] in degrees.
        """

        d = {
            'color': color('gray'),
            'stroke': 'none',
            'stroke-width': 0,
        }
        for p in pars: d[p] = pars[p]

        start = self._polar_to_cartesian(pos, r, a[0])
        end = self._polar_to_cartesian(pos, r, a[1])

        sweep = 1 if a[1] - a[0] <= 180 else 0
        self.t += '<path d="M%.1f %.1f L%.1f %.1f A%i %i 0 0 %i %.1f %.1f z" style="fill:%s; stroke-width: %i; stroke: %s"></path>' % (
            pos[0], pos[1], start[0], start[1], r, r, sweep, end[0], end[1], self._hex(d['color']), d['stroke-width'], self._hex(d['stroke']))

    def _polar_to_cartesian(self, pos, r, a):
        """Pass an angle in degrees to get out cartesian coordinates."""
        rad = (a - 90)*math.pi/180.0
        x = pos[0] + r*math.cos(rad)
        y = pos[1] + r*math.sin(rad)
        return (x, y)

    def graph_probability_matrix(self, singlevec, pairmat, stim='', singleframe=[], pairframe=[], pars={}):
        default = {
            'min-double-prob': -1,
            'spacing': 0,
            'square': (4, 4),
            'border': 'none',
            'offset-top': 20,
            'offset-left': 100,
            'offset-between': 4,
            'stimuli': {
                'plus': {'color': 'mint'},
                'minus': {'color': 'red'},
                'neutral': {'color': 'blue'},
                'other': {'color': 'gray'}
            },
        }
        for p in pars: default[p] = pars[p]

        y = default['offset-top']
        for i, s in enumerate(singlevec):
            x = default['offset-left'] + i*(default['square'][0] + default['spacing'])
            clr = self.colorscale3(s)
            clr = 'rgb(%i,%i,%i)' % (clr[0], clr[1], clr[2])
            self.square((x, y), default['square'], {'color':clr})

        y = default['offset-top'] + default['square'][1] + default['offset-between']
        for i in range(np.shape(pairmat)[0]):
            for j in range(np.shape(pairmat)[1]):
                x = default['offset-left'] + i*(default['square'][0] + default['spacing'])
                yy = y + j*(default['spacing'] + default['square'][1])
                clr = self.colorscale3(pairmat[i, j])
                clr = 'rgb(%i,%i,%i)' % (clr[0], clr[1], clr[2])
                self.square((x, yy), default['square'], {'color':clr})

        clr = 'gray' if len(stim) == 0 else default['stimuli'][stim]['color']
        self._markcells(singleframe, pairframe, color(clr), default)

    def _single_pos(self, i, pars):
        y = pars['offset-top']
        x = pars['offset-left'] + i*(pars['square'][0] + pars['spacing'])
        return (x, y)

    def _joint_pos(self, i, j, pars):
        y = pars['offset-top'] + pars['square'][1] + pars['offset-between']
        x = pars['offset-left'] + i*(pars['square'][0] + pars['spacing'])
        y += j*(pars['spacing'] + pars['square'][1])
        return (x, y)

    def _markcells(self, single, joint, clr, pars):
        for s in single:
            x, y = self._single_pos(s, pars)
            self.square((x, y), pars['square'], {'color': 'none',
                'stroke': clr,
                'stroke-width': 1,})

        for j in joint:
            x, y = self._joint_pos(j[0], j[1], pars)
            self.square((x, y), pars['square'], {'color': 'none',
                'stroke': clr,
                'stroke-width': 1,})


    def cellspots(self, centroids, rates, probs, active, match='', pars={}):
        default = {
            'imsize': (768, 512),
            'subsize': (300, 200),
            'spacing': 10,
            'cell-size': (2, 20),
            'cutoff': 0.1,
            'squarify': True,

            'stimuli': {
                'plus': {'color': 'mint', 'order':1},
                'minus': {'color': 'red', 'order':2},
                'neutral': {'color': 'blue', 'order':3},
            },
        }
        for p in pars: default[p] = pars[p]

        ssz, isz, csz = default['subsize'], default['imsize'], default['cell-size']
        default['scale'] = (float(ssz[0])/isz[0], float(ssz[1])/isz[1])
        default['cell-size'] = (csz[0], csz[1], float(csz[1] - csz[0]))

        if default['squarify']:
            centroids = self._squarify_centroids(centroids, default)
        else:
            centroids = [(c[0]*pars['scale'][0], c[1]*pars['scale'][1]) for c in centroids]

        roicount = self._count_rois(rates, active, [s for s in default['stimuli']], default['cutoff'])
        border = color(default['stimuli'][match]['color']) if match in default['stimuli'] else color('gray')
        self._uniscalestim((10, 10), centroids, rates, probs, active, roicount, border, default)


    def _uniscalestim(self, pos, centroids, rates, probs, active, roicount, border, pars):
        self.square(pos, pars['subsize'], {'color':'none',
            'stroke': border,
            'stroke-width': 2,
            })

        roin = np.zeros(len(roicount))
        stims = sorted([(pars['stimuli'][s]['order'], s) for s in pars['stimuli']])
        stims = [s[1] for s in stims]
        for k, stim in enumerate(stims):
            for i in range(len(centroids)):
                x = pos[0] + centroids[i][0]
                y = pos[1] + centroids[i][1]

                if k == 0:
                    self.circle((x, y), 1, {'color':'#CCCCCC'})

                if rates[stim][i] > pars['cutoff'] and active[i]:
                    r = rates[stim][i]*pars['cell-size'][2] + pars['cell-size'][0]
                    clr = self.colorscale2(probs[stim][i], pars['stimuli'][stim]['color'])
                    if roicount[i] > 1:
                        theta = 360.0/roicount[i]
                        start = -90 + theta*roin[i]
                        self.arc((x, y), r, (start, start + theta),
                            {'color':clr, 'stroke':color(pars['stimuli'][stim]['color']), 'stroke-width': 1})
                        roin[i] += 1

                    else:
                        self.circle((x, y), r, {'color':clr, 'stroke':color(pars['stimuli'][stim]['color']), 'stroke-width': 1})



    def _count_rois(self, rates, active, stims, cutoff):
        roicount = np.zeros(len(active))
        for stim in stims:
            for i in range(len(rates[stim])):
                if active[i] and rates[stim][i] > cutoff:
                    roicount[i] += 1

        return roicount

    def cellscale(self, centroids, rates, probs, active=[], pars={}):
        default = {
            'imsize': (768, 512),
            'subsize': (300, 200),
            'spacing': 10,
            'cell-size': (2, 20),
            'cutoff': 0.1,
            'squarify': True,

            'stimuli': {
                'plus': {'color': 'mint'},
                'minus': {'color': 'red'},
                'neutral': {'color': 'blue'},
            },
        }
        for p in pars: default[p] = pars[p]

        default['scale'] = (float(default['subsize'][0])/default['imsize'][0],
            float(default['subsize'][1])/default['imsize'][1])
        default['cell-size'] = (default['cell-size'][0], default['cell-size'][1],
            float(default['cell-size'][1] - default['cell-size'][0]))

        if default['squarify']:
            centroids = self._squarify_centroids(centroids, default)
        else:
            ocens = [(c[0]*pars['scale'][0], c[1]*pars['scale'][1]) for c in centroids]

        for i, stim in enumerate(default['stimuli']):
            y = default['spacing']
            x = default['spacing'] + i*(default['spacing'] + default['subsize'][0])
            self._scalestim((x, y), centroids, rates[stim], probs[stim], default['stimuli'][stim]['color'], active, default)

    def _squarify_centroids(self, cens, pars):
        area = (pars['subsize'][0]*pars['subsize'][0])/len(cens)
        side = math.sqrt(area)
        ncol = math.ceil(pars['subsize'][0]/side)
        side = pars['subsize'][0]/ncol
        nrow = math.floor(pars['subsize'][1]/side)

        while nrow*ncol < len(cens):
            xdiff = pars['subsize'][0] - ncol*side
            ydiff = pars['subsize'][1] - nrow*side

            if xdiff > ydiff:
                ncol += 1
                side = pars['subsize'][0]/ncol
                nrow = math.floor(pars['subsize'][1]/side)
            else:
                nrow += 1
                side = pars['subsize'][1]/nrow
                ncol = math.floor(pars['subsize'][0]/side)

        xdiff = pars['subsize'][0] - ncol*side
        ydiff = pars['subsize'][1] - nrow*side

        ocens = []
        for i in range(len(cens)):
            x = xdiff/2.0 + side/2.0 + (i%ncol)*side
            y = ydiff/2.0 + side/2.0 + math.floor(float(i)/ncol)*side
            ocens.append((x, y))

        return ocens

    def _scalestim(self, pos, centroids, rates, probs, maxclr, active, pars):
        self.square(pos, pars['subsize'], {'color':'none',
            'stroke': color(maxclr),
            'stroke-width': 2,
            })

        for i in range(len(centroids)):
            x = pos[0] + centroids[i][0]
            y = pos[1] + centroids[i][1]
            if rates[i] > pars['cutoff'] and (len(active) == 0 or active[i]):
                r = rates[i]*pars['cell-size'][2] + pars['cell-size'][0]
                clr = self.colorscale2(probs[i], maxclr)
                self.circle((x, y), r, {'color':clr, 'stroke':color('gray'), 'stroke-width': 1})
            else:
                self.circle((x, y), 1, {'color':'#CCCCCC'})



    def colorscale3(self, val):
        val = max(min(val, 1.0), 0.0)

        c1 = (0, 0, 0) # black
        c2 = (242, 226, 5) # white
        c3 = (200, 128, 209) # purple

        if val <= 0.5:
            diff = 2*val
            r, g, b = [(c2[i] - c1[i])*diff + c1[i] for i in range(3)]
            out = (int(round(r)), int(round(g)), int(round(b)))
        else:
            diff = 2*(val - 0.5)
            r, g, b = [(c3[i] - c2[i])*diff + c2[i] for i in range(3)]
            out = (int(round(r)), int(round(g)), int(round(b)))

        return out

    def colorscale2(self, val, maxc='purple'):
        val = max(min(val, 1.0), 0.0)

        #c1 = (0, 0, 0) # black
        c1 = (255, 255, 255) # white
        c2 = self._rgb(maxc) # purple

        r, g, b = [(c2[i] - c1[i])*val + c1[i] for i in range(3)]
        out = (int(round(r)), int(round(g)), int(round(b)))

        return out

    def _hex_to_rgb(self, val):
        """Convert hexidecimal color to rgb tuple."""
        val = val.lstrip('#')
        l = len(val)
        return tuple(int(val[i:i+l//3], 16) for i in range(0, l, l//3))

    def _rgb_to_hex(self, val):
        if val == 'none': return 'none'
        return '#%02x%02x%02x' % val

    def _rgb(self, val):
        """Convert string or hex to rgb tuple."""
        if isinstance(val, tuple): return val
        elif val[0] == '#': return self._hex_to_rgb(val)
        else: return self._hex_to_rgb(color(val))

    def _hex(self, val):
        if isinstance(val, tuple): return self._rgb_to_hex(val)
        elif val[0] == '#': return val
        else: return self._rgb_to_hex(self._rgb(val))

    def graph(self, save=''):
        """Finalize the graph and save if desired."""
        self.t += '</svg>'
        if len(save) > 0:
            if save[-3:].lower() == 'png':
                cairosvg.svg2png(bytestring=self.t, write_to=save)
            elif save[-3:].lower() == 'pdf':
                cairosvg.svg2pdf(bytestring=self.t, write_to=save)
            elif save[-3:].lower() == 'svg':
                fp = open(save, 'w')
                fp.write(self.t)
                fp.close()
        else:
            return self.t

def graph_probability_matrix(singlevec, pairmat, singleframe=[], pairframe=[], pars={}, save=''):
    svg = SVGraph()
    svg.replay(singlevec, pairmat, singleframe, pairframe, pars)
    return svg.graph(save)

def cellscale(centroids, rates, probs, active=[], pars={}, save=''):
    svg = SVGraph()
    svg.cellscale(centroids, rates, probs, active, pars)
    return svg.graph(save)

def cellspots(centroids, rates, probs, active, match='', pars={}, save=''):
    svg = SVGraph()
    svg.cellspots(centroids, rates, probs, active, match, pars)
    return svg.graph(save)

def svg(sz):
    out = SVGraph(sz)
    return out