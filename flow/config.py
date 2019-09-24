"""Configurable defaults for flow."""
from __future__ import print_function
from builtins import input
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
    'classifier': 'aode',  # 'naive-bayes' or 'aode'
    'stimulus-training-offset-ms': 50,
    'stimulus-training-ms': 950,  # Usually 1950
    'classification-ms': 260,  # usually 380, should be set to 190 based on recent data
    'excluded-time-around-onsets-ms': 2000,  # For training 'other' category

    # HOW TO DEAL WITH OTHER RUNNING
    'other-running-speed-threshold-cms': 4.0,  # Threshold in cm/s
    'other-running-fraction': 0.3,  # Fraction of other-running period required to be greater than threshold

    # ACCOUNT FOR DECREASED ACTIVITY IN SPONTANEOUS RUNS
    # Analog variables
    'analog-training-multiplier': 0.75,  # 0.35,  # Multiply deconvolved values by this before passing to classifier
    'analog-comparison-multiplier': 2.0,  # 1.5,  # Multiply deconvolved values by this before passing to classifier,
                                          # usually 1.5

    # TEMPORAL CLASSIFIER
    'temporal-dependent-priors': True,
    'temporal-prior-fwhm-ms': 260,  # in milliseconds, 3 frames FWHM
    'temporal-prior-baseline-activity': 0.0,  # Derived from looking at spontaneous period activity levels, subtracted
    'temporal-prior-baseline-sigma': 3.0,  # Maximum amount of variance explained by noise, divided by
    'temporal-prior-threshold': -1,  # Set a threshold on the temporal prior so that frames are either analyzed or not
    'remove-stim': True,  # Remove the stimulus frames before temporal prior and classification

    # ACCOUNT FOR LICKING
    # Cut off training trials with numbers of licks during the
    # lick-window > lick-cutoff if lick-cutoff is >= 0.
    'lick-cutoff': -1,  # Number of licks required before cutoff?
    'lick-window': (-1, 0),  # Window in which licking is identified

    # ACCOUNT FOR DIFFERENT NUMBERS OF STIMULUS PRESENTATIONS
    'equalize-stimulus-onsets': False,  # UNIMPLEMENTED Train on the same number of onsets of each stimulus
    'maximum-cs-onsets': -1,  # Maximum number of CS onsets to train on
    'train-only-on-positives': False,  # On correct rejections and true positives (cs+ lick, csn, cs- no lick)

    # RANDOMIZATION
    'equalize-cell-activity': False,  # If true, equalize the activity of each cell across all runs

    'classifier-updated-date': '190405',

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

_psytrack_defaults = {
    'weights': {
        'bias': 1,
        'ori_0': 1,
        'ori_135': 1,
        'ori_270': 1,
        'prev_choice': 1,
        'prev_reward': 1,
        'prev_punish': 1},
    'include_pavlovian': False,
    'separate_day_var': True,
    'fixed_sigma': None,
    'fixed_sigma_day': None,
    'updated': 190812,  # 190802,  # 190715,  # 190620,  # The current update date for the PsyTracker code
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


def session_parameter(key, val):
    """
    Overwrite a parameter only for the current session.

    Parameters
    ----------
    key : str
        Name of a parameter
    val : variable
        New value of the parameter

    """

    global _params
    if _params is None:
        _params = _load_config()

    _params[key] = val


def reconfigure():
    """Re-set user-configurable parameters."""
    config_path = _find_config()

    print("Reconfiguring flow: {}".format(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("PATHS")
    if 'paths' not in config:
        config['paths'] = {}
    data_path = input(
        'Enter path to data: [{}] '.format(config['paths'].get('data', '')))
    if len(data_path):
        config['paths']['data'] = os.path.normpath(data_path)

    output_path = input(
        'Enter path to analyzed output files: [{}] '.format(
            config['paths'].get('output', '')))
    if len(output_path):
        config['paths']['output'] = os.path.normpath(output_path)

    graph_path = input(
        'Enter path to graphing directory: [{}] '.format(
            config['paths'].get('graph', '')))
    if len(graph_path):
        config['paths']['graph'] = os.path.normpath(graph_path)

    metadata_path = input(
        'Enter path to metadata json file: [{}] '.format(
            config['paths'].get('metadata', '')))
    if len(metadata_path):
        config['paths']['metadata'] = os.path.normpath(metadata_path)

    with open(config_path, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4, separators=(',', ': '))

    params(reload_=True)


def default():
    """Return default parameters."""
    p = params()
    return p['defaults']


def _load_config():
    config_path = _find_config()
    with open(config_path, 'r') as f:
        loaded_config = json.load(f)
        config = {
            'defaults': copy.copy(_settings),
            'psytrack_defaults': copy.copy(_psytrack_defaults)}
        for key in loaded_config:
            # Just add keys in the config file other than 'defaults'
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
    config_path = _initialize_config()
    return config_path


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
        except IOError:
            continue
        else:
            config_path = os.path.join(path, CONFIG_FILE)
            print("Configuration initialized to: {}".format(config_path))
            print("Run `import flow.config as cfg; cfg.reconfigure()` " +
                  "to update.")
            return config_path
    print("Unable to find writable location.")
    return DEFAULT_FILE


if __name__ == '__main__':
    params()
# from pudb import set_trace; set_trace()
