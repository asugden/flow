"""Configurable defaults for flow."""
import copy
import json
import os
import shutil

CONFIG_FILE = 'flow.cfg'
DEFAULT_FILE = 'flow.cfg.default'
CONFIG_PATHS = [
    os.path.expanduser('~/.config/flow'),
    os.path.join(os.path.dirname(__file__)),
    os.environ.get("FLOW_CONF"),
]


_settings = {
    # MOUSE
    'mouse': '',
    'training-date': '',
    'training-runs': [],
    'comparison-date': '',  # UNIMPLEMENTED-- must equal training-date
    'comparison-run': 0,
    'training-other-running-runs': [1, 5],

    # CLASSIFIER
    'classifier': 'aode-analog',  # 'naive-bayes' or 'aode-analog'
    'stimulus-training-offset-ms': 50,
    'stimulus-training-ms': 950,  # Usually 1950
    'classification-ms': 260,  # usually 380, should be set to 190 based on recent data
    'excluded-time-around-onsets-ms': 2000,  # For training 'other' category

    # HOW TO DEAL WITH OTHER RUNNING
    'other-running-speed-threshold-cms': 4.0,  # Threshold in cm/s
    'other-running-fraction': 0.3,  # Fraction of other-running period required to be greater than threshold

    # ACCOUNT FOR DECREASED ACTIVITY IN SPONTANEOUS RUNS
    # Analog variables
    'analog-training-multiplier': 0.35,  # 0.35,  # Multiply deconvolved values by this before passing to classifier
    'analog-comparison-multiplier': 1.5,  # 1.5,  # Multiply deconvolved values by this before passing to classifier,
                                          # usually 1.5

    # TEMPORAL CLASSIFIER
    'temporal-dependent-priors': True,
    'temporal-prior-fwhm-ms': 260,  # in milliseconds, 3 frames FWHM
    'temporal-prior-baseline-activity': 0.0,  # Derived from looking at spontaneous period activity levels, subtracted
    'temporal-prior-baseline-sigma': 3.0,  # Maximum amount of variance explained by noise, divided by
    'temporal-prior-threshold': -1,  # Set a threshold on the temporal prior so that frames are either analyzed or not

    # ACCOUNT FOR LICKING
    # Cut off training trials with numbers of licks during the
    # lick-window > lick-cutoff if lick-cutoff is >= 0.
    'lick-cutoff': -1,  # Number of licks required before cutoff?
    'lick-window': (-1, 0),  # Window in which licking is identified

    # ACCOUNT FOR DIFFERENT NUMBERS OF STIMULUS PRESENTATIONS
    'equalize-stimulus-onsets': False,  # UNIMPLEMENTED Train on the same number of onsets of each stimulus
    'maximum-cs-onsets': -1,  # Maximum number of CS onsets to train on
    'train-only-on-positives': False,  # On correct rejections and true positives (cs+ lick, csn, cs- no lick)

    'classifier-updated-date': '180516',  # Advance one day to 180517 for naive bayes

    # ==================================================================
    # LEAVE UNCHANGED AFTER TRAINING

    'trace-type': 'deconvolved',  # Previously used dff, have switched entirely to deconvolved

    # PRIORS- TUNED BY HAND
    # Priors are dependent on the classifier
    'probability': {
        'plus': 0.05,  # usually 0.05
        'minus': 0.05,  # usually 0.05,
        'neutral': 0.05,  # usually 0.05
        # 'ensure': 0.05,
        # 'quinine': 0.05,
        'plus-neutral': 0.05,
        'plus-minus': 0.05,
        'neutral-minus': 0.05,
        'other': 0.997,  # usually 0.997
        'other-running': 0.997,  # usually 0.997
        'disengaged1': 0.05,
        'disengaged2': 0.05,
    },

    # TRAINING EXTRAS
    # Remove to have pavlovians not included in CS plus training
    'training-equivalent': {
        'blank': 'other',
        'pavlovian': 'plus',
    }
}

_colors = {
    'plus': '#47D1A8',  # mint
    'neutral': '#47AEED',  # blue
    'minus': '#D61E21',  # red
    'plus-neutral': '#8CD7DB',  # aqua
    'plus-minus': '#F2E205',  # yellow
    'neutral-minus': '#C880D1',  # purple
    'other': '#7C7C7C',  # gray
    'other-running': '#333333',  # dark gray
    'pavlovian': '#47D1A8',  # green
    'real': '#47D1A8',  # yellow
    'circshift': '#8CD7DB',  # aqua
    'run-onset': '#C880D1',  # purple
    'motion-onset': '#D61E21',  # red
    'popact': '#333333',  # dark gray
    'disengaged1': '#F2E205',  # yellow
    'disengaged2': '#E86E0A',  # orange
    'ensure': '#5E5AE6',  # indigo
    'quinine': '#E86E0A',  # orange

    'plus-only': '#47D1A8',  # mint
    'neutral-only': '#47AEED',  # blue
    'minus-only': '#D61E21',  # red
    'ensure-only': '#5E5AE6',  # indigo
    'quinine-only': '#E86E0A',  # orange
    'ensure-multiplexed': '#5E5AE6',  # indigo
    'quinine-multiplexed': '#E86E0A',  # orange
    'plus-ensure': '#5E5AE6',  # indigo
    'minus-quinine': '#E86E0A',  # orange

    'lick': '#F2E205',  # yellow
    'undefined': '#7C7C7C',  # gray
    'multiplexed': '#000000',  # black
    'combined': '#000000',  # black
    'temporal-prior': '#C880D1',  # purple

    'inhibited': '#7C7C7C',  # gray

    'reward-cluster-1': '#5E5AE6',  # indigo
    'reward-cluster-exclusive-1': '#C880D1',  # purple
}

_params = None


def params(reload_=False):
    """Return a copy of the parameters dictionary.

    This is the primary function that should be used to access user-specific
    parameters.

    For 'defaults' and 'colors', they are initialized with the default values
    in this file, but overwritten by any settings in the user's config file.

    Parameters
    ----------
    reload_ : bool
        If True, reload the config file from disk.

    """
    global _params
    if reload_ or _params is None:
        _params = _load_config()
    return copy.deepcopy(_params)


def reconfigure():
    """Re-set user-configurable parameters."""
    config_path = _find_config()

    print("Reconfiguring flow: {}".format(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("PATHS")
    data_path = raw_input(
        'Enter path to data: [{}] '.format(config['paths']['data']))
    if len(data_path):
        config['paths']['data'] = os.path.normpath(data_path)

    output_path = raw_input(
        'Enter path to analyzed output files: [{}] '.format(
            config['paths']['output']))
    if len(output_path):
        config['paths']['output'] = os.path.normpath(output_path)

    graph_path = raw_input(
        'Enter path to graphing directory: [{}] '.format(
            config['paths']['graph']))
    if len(output_path):
        config['paths']['graph'] = os.path.normpath(graph_path)

    print('ANALYSIS BACKEND')
    print(' memory: stores values in memory, not persistent')
    print(' shelve: stores values in a shelve database file')
    print(' couch: store values in a CouchDB (should already be running)')

    backend = None
    while backend not in config['backends']['supported_backends']:
        backend = raw_input(
            'Enter backend type: [{}] '.format(
                config['backends']['backend']))
        if not len(backend):
            backend = config['backends']['backend']
    config['backends']['backend'] = backend

    if backend == 'couch':
        if 'couch_options' not in config['backends']:
            config['backends']['couch_options'] = {}
        host = raw_input("Enter ip or hostname of CouchDB: [{}] ".format(
            config['backends']['couch_options'].get('host', None)))
        if len(host):
            config['backends']['couch_options']['host'] = host
        port = raw_input("Enter port for CouchDB: [{}] ".format(
            config['backends']['couch_options'].get('port', None)))
        if len(port):
            config['backends']['couch_options']['port'] = port
        database = raw_input("Enter name of analysis database: [{}] ".format(
            config['backends']['couch_options'].get('database', None)))
        if len(database):
            config['backends']['couch_options']['database'] = database
        user = raw_input("Enter username to authenticate with CouchDB (optional): [{}] ".format(
            config['backends']['couch_options'].get('user', None)))
        if len(user):
            config['backends']['couch_options']['user'] = user
        password = raw_input("Enter password to authenticate with CouchDB (optional): [{}] ".format(
            config['backends']['couch_options'].get('password', None)))
        if len(password):
            config['backends']['couch_options']['password'] = password

    with open(config_path, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ': '))

    params(reload_=True)


def defaults():
    """Return default parameters."""
    p = params()
    return p['defaults']


def colors(clr=None):
    """Return default color pairings.

    Parameters
    ----------
    clr : str, optional
        If not None, return the default colors for a specific group.
        Otherwise return all color pairings.

    """
    p = params()
    if colors is None:
        return p['colors']
    else:
        return p['colors'].get(clr, '#7C7C7C')


def _load_config():
    global _settings, _colors
    config_path = _find_config()
    if config_path is None:
        config_path = _initialize_config()
        print("Configuration initialized to: {}".format(config_path))
        print("Run `import flow.config as cfg; cfg.reconfigure()` to update.")
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
        config = {'defaults': copy.copy(_settings),
                  'colors': copy.copy(_colors)}
        for key in loaded_config:
            # Just add keys in the config file other than
            # 'defaults' and 'colors'
            if key not in config:
                config[key] = loaded_config[key]
            else:
                # This assumes that these keys will also contain dicts,
                # they should.
                config[key].update(loaded_config[key])
    return config


def _find_config():
    for path in CONFIG_PATHS:
        if path is None:
            continue
        if os.path.isfile(os.path.join(path, CONFIG_FILE)):
            return os.path.join(path, CONFIG_FILE)
    return None


def _initialize_config():
    for path in CONFIG_PATHS:
        if path is None:
            continue
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except OSError:
                continue
        f = os.path.join(os.path.dirname(__file__), DEFAULT_FILE)
        try:
            shutil.copy(f, os.path.join(path, CONFIG_FILE))
            return os.path.join(path, CONFIG_FILE)
        except IOError:
            continue
    print("Unable to find writable location.")
    return DEFAULT_FILE


if __name__ == '__main__':
    params()
# from pudb import set_trace; set_trace()
