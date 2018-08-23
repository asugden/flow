# Should this be deprecated?
# PARSE.py
# Parse common inputs. Used by metadata and Physalize
# Last updated 2/21/14

import re
from math import ceil

class Parser():
    def __init__(self):
        pass

    # CLI KEY VALUES
    # Split by -key or --key and values or ranges
    def keyvals(self, kv, comparators=False):
        args = filter(lambda x: False if x.strip() == '' or x.strip() == '-' else True, re.split('((^| )-(?=[a-z]))', kv.strip()))
        out = {}
        for argf in args:
            arg = filter(lambda x: False if x == '' or x == ' ' or x == '=' or x == '"' or x == "'" else True, re.split('( |<|>|==|=|\'|\")', argf))
            arg = [a.strip() for a in arg]

            if comparators:
                if len(arg) == 1: out[self.str(arg[0]).lower()] = ('=', True)
                elif arg[1][0] == '>': out[self.str(arg[0]).lower()] = ('>', self.unknown(arg[1][1:]))
                elif arg[1][0] == '<': out[self.str(arg[0]).lower()] = ('<', self.unknown(arg[1][1:]))
                elif arg[1][0:2] == '!=': out[self.str(arg[0]).lower()] = ('!', self.unknown(arg[1][2:]))
                elif arg[1][0] == '!': out[self.str(arg[0]).lower()] = ('!', self.unknown(arg[1][1:]))
                else: out[self.str(arg[0]).lower()] = ('=', self.unknown(arg[1]))
            else:
                if len(arg) == 1: out[self.str(arg[0]).lower()] = True
                elif len(arg) > 2:
                    out[self.str(arg[0]).lower()] = [self.unknown(v) for v in arg]
                else:
                    out[self.str(arg[0]).lower()] = self.unknown(arg[1])
        return out

    # CLEAN ARGUMENTS
    # Takes two dictionaries, a user-entered dict and a default dict, adds all user key-
    # val pairs to the default dict and returns it. If vals of user dict don't match
    # possible input values from defaults, they are skipped if nonvals = False. If keys
    # are not in defaults, they are skipped if nonkeys = False
    # Treats None types as ints
    def cleanargs(self, kv, defaults, nonkeys=False, nonvals=False):
        out = {}
        # Set defaults of the argument
        for key in defaults:
            if type(defaults[key]) == list:
                out[key.lower()] = defaults[key][0]
            else: out[key.lower()] = defaults[key]

        for key in kv:
            key = key.lower()
            if key in defaults:
                if nonvals or (isinstance(defaults[key], list) and kv[key] in defaults[key]) or (self.isnum(defaults[key]) and self.isnum(kv[key])) or (self.isdate(defaults[key]) and self.isdate(kv[key])) or (isinstance(defaults[key], str) and isinstance(kv[key], str)) or (self.isnum(kv[key]) and defaults[key] == None):
                    out[key] = kv[key]
            elif nonkeys:
                out[key] = kv[key]

        return out

    # PRINTABLE COLUMNS
    # Turn a list of values into printable columns and print them
    # Although parser is primarily input-based, many modules that call parser need access
    # to an appropriate way to print
    # l is list of values to print, title is optional title, width is the total printed
    # width and truncate trims each value to a fixed length, inset is number of tabs
    def columns(self, l, title='', width=80, truncate=-1, inset=0):
        if len(l) < 1: return False
        if len(title) > 0: print('\t'*inset + '%s:' % (title))

        width -= 5 # for the sake of tabs

        l.sort()
        l = [self.stringify(ll) for ll in l]
        column_width = max(len(word) for word in l) + 2  # padding
        if truncate > 0: column_width = min(truncate + 2, column_width)

        ncolumns = width/column_width if column_width < width else 1
        rowified = [[l[i + ncolumns*j] for i in range(ncolumns) if len(l) > i + ncolumns*j] for j in range(int(ceil(float(len(l))/ncolumns) + 0.5))]

        for row in rowified:
            print('\t'*(inset + 1) + ''.join(word[:column_width].ljust(column_width) for word in row))

    # INTERNAL METHOD: STRINGIFY
    # Convert to string, return type, and force everything into 4 sig figs
    def stringify(self, s, vtype='unknown'):
        if isinstance(s, list) or isinstance(s, tuple):
            aout = []
            rtype = ''
            for el in s:
                nel, vtype = self.stringify(el, vtype)
                rtype = vtype if rtype == '' or rtype == 'int' and vtype == 'float' or vtype == 'str' else rtype
                aout.append(nel)
            return '[' + ', '.join(aout) + ']'
        elif isinstance(s, float) or isinstance(s, int):
            if vtype == 'unknown': vtype = 'float' if isinstance(s, float) else 'int'
            elif vtype == 'int' and isinstance(s, float): vtype = 'float'
            return '%.4g' % (s)
        else:
            if vtype == 'unknown': vtype = 'str'
            return s.strip()

    # DATE OF FILE
    # Get date by name of file, file number as optional second
    def datefile(self, file):
        dt = self.filename(file).replace('-', '_').split('_')
        return ((int(dt[0])%2000, int(dt[1]), int(dt[2])), int(dt[3]))

    # CELL NAME BY PATH
    # Get the date and

    # FILE NAME FROM PATH
    # Get name from path or return if already name
    def filename(self, path):
        end = path.rfind('.') if path.rfind('.') > 0 else len(path)
        return path[path.rfind('/') + 1:end]

    # STRINGIFY DATE
    # Convert date to 6-string
    def strdate(self, d):
        return '{:02g}{:02g}{:02g}'.format(d[0], d[1], d[2])

    # DATE
    # Return date as tuple of (year, month, day)
    # Takes input of mm/dd/yy or mm-dd-yy or yymmdd or yy mm dd
    def date(self, dp):
        dp = re.sub('[^\d\/ -]', '', dp.strip())
        ds = [
            '(^| |=)[0-9]{2,4} [0-9]{1,2} [0-9]{1,2}( |$)',
            '(^| |=)[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}( |$)',
            '(^| |=)[0-9]{1,2}-[0-9]{1,2}-[0-9]{1,2}( |$)',
            '(^| |=)[0-9]{6}( |$)',
        ]

        if re.match(ds[3], dp):
            out = re.search(ds[3], dp).group(0)
            out = [out[0:2], out[2:4], out[4:6]]
        elif re.match(ds[0], dp):
            out = re.search(ds[0], dp).group(0)
            out = out.split(' ')
        elif re.match(ds[1], dp):
            out = re.search(ds[1], dp).group(0)
            out = out.split('/')
            out = [out[2], out[0], out[1]]
        elif re.match(ds[2], dp):
            out = re.search(ds[2], dp).group(0)
            out = out.split('-')
            out = [out[2], out[0], out[1]]
        else:
            out = []

        return tuple([int(i) for i in out])

    # KEY VAL LIST
    # Split member by tabs
    def keyvallist(self, l):
        l = filter(lambda x: True if x != '' else False, l.strip().split('\t'))
        pars = {}
        for i in range(1, len(l)):
            p = l[i].split(' ')
            if len(p) > 1: pars[p[0].strip().lower()] = self.unknown(p[1])
            else: pars[p[0].strip().lower()] = True
        return (l[0], pars)

    # LIST
    # Split by explicit list
    def list(self, l, splitby=','):
        l = filter(lambda x: True if x != '' else False, l.strip().split(splitby))
        for i in range(len(l)): l[i] = self.unknown(l[i])
        return l

    # UNKNOWN TYPE
    # Parse as int if int, float if float, else str
    def unknown(self, u):
        u = u.strip()
        if re.match(self._getreg('float-range'), u): return self.float_range(u[1:-1])
        elif u.find(',') > -1: return self.list(u)
        elif re.match(self._getreg('int'), u): return self.int(u)
        elif re.match(self._getreg('float'), u): return self.float(u)
        elif re.match(self._getreg('bool'), u): return True if u[0].lower() == 't' else False
        elif len(self.date(u.lower())) == 3: return self.date(u)
        elif len(self.date_range(u.lower())) == 3: return self.date_range(u)
        else: return u

    # STRING
    # Strip whitespace
    def str(self, s):
        return s.strip()

    # INTEGER
    # Strip non-integer characters and convert to int
    def int(self, i):
        i = re.sub('[^\d-]', '', i)
        return int(i)

    # BOOLEAN
    # Return True if int > 0 or has 't' else False
    def bool(self, i):
        i = i.strip().lower()
        if re.match('^(-)?[\d]+$', i): return True if int(i) > 0 else False
        else: return True if i.find('t') > -1 else False

    # FLOATING POINT
    # Strip non-float characters and convert to float
    def float(self, f):
        f = re.sub('[^\d.-]', '', f)
        return float(f)

    # DATE RANGE
    # Fancier version of date
    # Accepts
    def date_range(self, dp):
        dp = re.sub('[^\d\/ -]', '',    dp.strip())
        ds = [
            '(^| |=)[0-9]{2,4}(-[0-9]{2,4})? [0-9]{1,2}(-[0-9]{1,2})? [0-9]{1,2}(-[0-9]{1,2})?( |$)',
            '(^| |=)[0-9]{1,2}(-[0-9]{1,2})?/[0-9]{1,2}(-[0-9]{1,2})?/[0-9]{2,4}(-[0-9]{2,4})?( |$)',
            '(^| |=)[0-9]{1,2}-[0-9]{1,2}-[0-9]{1,2}( |$)',
            '(^| |=)[0-9]{6}( |$)',
        ]

        if re.match(ds[2], dp) or re.match(ds[3], dp): return self.date(dp)
        elif re.match(ds[0], dp):
            out = re.search(ds[0], dp).group(0)
            out = out.split(' ')
        elif re.match(ds[1], dp):
            out = re.search(ds[1], dp).group(0)
            out = out.split('/')
            out = [out[2], out[0], out[1]]
        else:
            out = []

        return tuple([[int(j) for j in i.split('-')] for i in out])

    # STRING RANGE
    # Split by |
    def str_range(self, s):
        s = s.split('|')
        return [self.str(i) for i in s]

    # INTEGER RANGE
    # Split and convert. Cannot do hyphen due to negative numbers
    def int_range(self, i):
        i = i.split(',')
        return [self.int(j) for j in i]

    # FLOATING POINT RANGE
    # Split and convert
    def float_range(self, f):
        f = f.split(',')
        return [self.float(i) for i in f]


    # CHECK IF INPUT IS A NUMBER
    def isnum(self, txt): return self.isint(txt) or self.isfloat(txt)

    # TYPE INT OR INT STR
    def isint(self, txt):
        if isinstance(txt, str): return re.match(self._getreg('int'), txt.strip())
        if isinstance(txt, int): return True
        return False

    def isfloat(self, txt):
        if isinstance(txt, str): return re.match(self._getreg('float'), txt.strip())
        if isinstance(txt, float): return True
        return False

    def isdate(self, txt):
        if isinstance(txt, tuple) or isinstance(txt, list) and len(list) == 3 and isinstance(txt[0], int) and isinstance(txt[1], int) and isinstance(txt[2], int):
            yr = True if (txt[0] >= 0 and txt[0] < 100) or (txt[0] > 1980 and txt[0] < 2100) else False
            mn = True if (txt[1] > 0 and txt[1] < 13) else False
            dy = True if (txt[2] > 0 and txt[2] < 32) else False
            return yr and mn and dy
        if isinstance(txt, str):
            dtypes = ['space', 'slash', 'dash', '6d', '8d']
            for t in dtypes:
                if re.match(self._getreg('date-%s' % (t)), txt.strip()):
                    return True
        if isinstance(txt, int):
            txt = str(txt)
            if len(txt) > 3 and len(txt) < 7: return self.isdate(txt.zfill(6))
            if len(txt) == 8: return self.isdate(txt)

        return False


    # GET COMPILED REGULAR EXPRESSION
    def _getreg(self, r):
        if not hasattr(self, '_compiled_regs'):
            self._compiled_regs = {
                'key': '((^| )-(?=[a-z]))', # Split keyvals by key
                'val': '( |=)', # Separate keys from vals
                'date-space': '(^| |=)[0-9]{2,4} [0-9]{1,2} [0-9]{1,2}( |$)', # YY[YY] M[M] D[D]
                'date-slash': '(^| |=)[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}( |$)', # M[M]/D[D]/YY[YY]
                'date-dash': '(^| |=)[0-9]{1,2}-[0-9]{1,2}-[0-9]{1,2}( |$)', # M[M]-D[D]/M
                'date-6d': '(^| |=)[0-9][0-9][0-1][0-9][0-3][0-9]( |$)', # Six digit date string YYMMDD
                'date-8d': '(^| |=)[1-2][0,9][0-9][0-9][0-1][0-9][0-3][0-9]( |$)', # Eight digit date string YYYYMMDD
                'float': '^(-)?[\d.]+$',
                'int': '^(-)?[\d]+$',
                'float-range': '^\[[\s\d.-]+,[\s\d.-]+\]$',
                'bool': '^(true|false|t|f)$',
            }

            # Compile all regular expressions
            for key in self._compiled_regs: self._compiled_regs[key] = re.compile(self._compiled_regs[key], flags=re.IGNORECASE)

        return self._compiled_regs[r] if r in self._compiled_regs else False
