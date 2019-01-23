from builtins import object, range, str

import numpy as np

"""
Save parameters in a legible text format that can easily be read back in
"""


class Metadata(object):
    def __init__(self, path):
        self.path = path

    # Add a single analysis file to protocols
    def read(self):
        fp = open(self.path, 'r')
        lines = fp.read().split('\n')
        fp.close()

        out = {}
        for line in lines:
            if line.find(':') > -1:
                line = line.strip().split(':')
                key = line[0][:line[0].find('[')]
                type = line[0].strip()[line[0].find('[')+1:-1]
                out[key] = self.metadata_type(type, line[1].strip())

        return out

    # Set the metdata to type, taking into account lists or tuples
    def metadata_type(self, type, val):
        if (val[:2] == '[[' and val[-2:] == ']]'):
            out = []
            vals = val[2:-2].replace('], [', '][').split('][')
            for l in vals:
                out.append([])
                l = l.split(',')
                for v in l: out[-1].append(int(v) if type == 'int' else float(v) if type == 'float' else str(v))
            return out
        elif (val[0] == '[' and val[-1] == ']') or (val[0] == '(' and val[-1] == ')'):
            out = []
            vals = val[1:-1].split(',')
            for v in vals: out.append(int(v) if type == 'int' else float(v) if type == 'float' else str(v))
            if val[0] == '(': return tuple(out)
            else: return out
        else:
            return int(val) if type == 'int' else float(val) if type == 'float' else str(val)

    # ==================================================================================
    # SAVING VARIABLES

    # Write files to a directory based on analysis type (first name before dash) and experiment name in protocol
    def write(self, pars):
        fo = open(self.path, 'w')
        for val in pars: # val is protocol key to be saved
            if isinstance(pars[val], dict):
                fo.write('{:>24s}:{{\n'.format('%s[dict]' % (val)))
                for vval in pars[val]:
                    if isinstance(pars[val][vval], dict):
                        fo.write('{:>28s}:{{\n'.format('%s[dict]' % (vval)))
                        for vvval in pars[val][vval]:
                            out, type = self.stringify(pars[val][vval][vvval])
                            fo.write('{:>32s}:{:s}\n'.format('%s[%s]' % (vvval, type), out))
                        fo.write('{:>29s}}}\n'.format(' '))
                    else:   
                        out, type = self.stringify(pars[val][vval])
                        fo.write('{:>28s}:{:s}\n'.format('%s[%s]' % (vval, type), out))
                fo.write('{:>25s}}}\n'.format(' '))
            else:
                out, type = self.stringify(pars[val])
                fo.write('{:>24s}:{:s}\n'.format('%s[%s]' % (val, type), out))
        fo.close()

    # Convert val to string, return type, and force everything into 4 sig figs
    def stringify(self, s, vtype='unknown'):
        s = self.listify(s)
        if isinstance(s, list) or isinstance(s, tuple):
            aout = []
            rtype = ''
            for el in s:
                nel, vtype = self.stringify(el, vtype)
                rtype = vtype if rtype == '' or rtype == 'int' and vtype == 'float' or vtype == 'str' else rtype
                aout.append(nel)
            return ('[' + ', '.join(aout) + ']', vtype)
        elif isinstance(s, float) or isinstance(s, int):
            if vtype == 'unknown': vtype = 'float' if isinstance(s, float) else 'int'
            elif vtype == 'int' and isinstance(s, float): vtype = 'float'
            return ('%.4g' % (s), vtype)
        elif s is None:
            raise ValueError('Analysis module returned values of None.')
        else:
            if vtype == 'unknown': vtype = 'str'
            return (s.strip(), vtype)

    # Turn all numpy arrays into lists
    def listify(self, val):
        if type(val).__module__ == np.__name__: val = val.tolist()
        elif isinstance(val, list):
            for i in range(len(val)):
                val[i] = self.listify(val[i])
        return val

def read(path):
    out = Metadata(path)
    return out.read()

def write(path, pars):
    out = Metadata(path)
    out.write(pars)
