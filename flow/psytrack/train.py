import numpy as np
from pprint import pprint

from psytrack.aux.invBlkTriDiag import getCredibleInterval
from psytrack.hyperOpt import hyperOpt


def train(
        mouse,
        run_types=('training',), tags=('hungry',), exclude_tags=('bad',),
        weights=None,
        include_pavlovian=True,
        separate_day_var=True, verbose=False):
    """Main function use to train the PsyTracker."""

    if weights is None:
        weights = {
            'bias': 1,
            'ori_0': 1,
            'ori_135': 1,
            'ori_270': 1,
            'prev_choice': 1,
            'prev_reward': 1,
            'prev_punish': 1}

    k = np.sum([weights[i] for i in weights.keys()])
    if verbose:
        print('* Beginning training for: {} *'.format(mouse))
        print(' Fitting weights:')
        pprint(weights)
        print(' Fitting {} total hyper-parameters'.format(k))

    # Initialize hyperparameters
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*k}
    opt_list = ['sigma']

    # Add in parameters for each day if needed
    if separate_day_var:
        hyper['sigDay'] = [2**-4.]*k
        opt_list.append('sigDay')

    # Extract data from our simpcells and convert to format for PsyTrack
    if verbose:
        print('- Collecting data')
    data = _gather_data(
        mouse, run_types=run_types, tags=tags, exclude_tags=exclude_tags,
        include_pavlovian=include_pavlovian, weights=weights)
    if verbose:
        print(' Data keys:\n  {}'.format(sorted(data.keys())))
        print(' Inputs:\n  {}'.format(sorted(data['inputs'].keys())))
        print(' Total trials:\n  {}'.format(len(data['y'])))

    # Fit model
    if verbose:
        print('- Fitting model')
    hyp, evd, wMode, hess = hyperOpt(
        data, hyper, weights, opt_list,
        showOpt=0 if not verbose else int(verbose) - 1)

    # Calculate confidence intervals
    if verbose:
        print('- Determining confidence intervals')
    credible_int = getCredibleInterval(hess)

    results = {'hyper': hyp,
               'evidence': evd,
               'model_weights': wMode,
               'hessian': hess,
               'credible_intervals': credible_int}

    return data, results


def _parse_weights(weights):
    """Parse the weights dict to determine what data needs to be collected."""
    oris = []
    for key in weights:
        if key[:4] == 'ori_':
            oris.append(int(key[4:]))

    return oris


def _gather_data(
        mouse, weights, run_types, tags=None, exclude_tags=('bad',),
        include_pavlovian=True):
    orientations = _parse_weights(weights)
    day_length = []  # Number of trials for each day
    days = []
    y = []  # 2 for lick, 1 for no lick
    correct = []  # correct trials, boolean
    answer = []  # The correct choice, 2 for lick, 1 for no lick

    if 'prev_reward' in weights or 'cum_reward' in weights:
        reward = []  # rewarded trials, boolean
    if 'prev_punish' in weights or 'cum_punish' in weights:
        punish = []  # punished trials, boolean
    if 'cum_reward' in weights:
        cum_reward = []  # cumulative number of rewarded trials per day
    if 'cum_punish' in weights:
        cum_punish = []  # cumulative number of punish trials per day
    oris = {ori: [] for ori in orientations}
    for date in mouse.dates():
        date_ntrials = 0
        date_reward, date_punish = [], []
        for run in date.runs(
                run_types=run_types, tags=tags, exclude_tags=exclude_tags):
            t2p = run.trace2p()
            ntrials = t2p.ntrials
            if not include_pavlovian:
                raise NotImplementedError
            if not ntrials > 0:
                continue
            date_ntrials += ntrials

            run_choice = t2p.choice()
            assert(len(run_choice) == ntrials)
            y.extend(run_choice)

            run_errs = t2p.errors()
            assert(len(~run_errs) == ntrials)
            correct.extend(~run_errs)

            run_answer = np.logical_xor(
                run_choice, run_errs)  # This should be the correct action
            assert(len(run_answer) == ntrials)
            answer.extend(run_answer)

            if 'prev_reward' in weights or 'cum_reward' in weights:
                run_rew = t2p.reward() > 0
                assert(len(run_rew) == ntrials)
                date_reward.extend(run_rew)

            if 'prev_punish' in weights or 'cum_punish' in weights:
                run_punish = t2p.punishment() > 0
                assert(len(run_punish) == ntrials)
                date_punish.extend(run_punish)

            for ori in oris:
                ori_trials = [o == ori for o in t2p.orientations]
                assert(len(ori_trials) == ntrials)
                oris[ori].extend(ori_trials)

        if date_ntrials > 0:
            day_length.append(date_ntrials)
            days.append(date.date)

            reward.extend(date_reward)
            punish.extend(date_punish)

            if 'cum_reward' in weights:
                cum_reward.extend(np.cumsum(date_reward))
            if 'cum_punish' in weights:
                cum_punish.extend(np.cumsum(date_punish))

    out = {'mouse': mouse.mouse}
    out['dayLength'] = np.array(day_length)
    out['days'] = np.array(days)
    out['y'] = np.array([2 if val else 1 for val in y])
    out['correct'] = np.array(correct)
    out['answer'] = np.array([2 if val else 1 for val in answer])

    out['inputs'] = {}
    for ori in oris:
        key = 'ori_{}'.format(ori)
        ori_arr = np.array(oris[ori])
        out['inputs'][key] = np.ones((len(oris[ori]), 2))
        out['inputs'][key][:, 0] += ori_arr
        out['inputs'][key][1:, 1] += ori_arr[:-1]

    if 'prev_choice' in weights:
        out['inputs']['prev_choice'] = np.ones((len(out['y']), 2))
        out['inputs']['prev_choice'][1:, 0] = out['y'][:-1]
        out['inputs']['prev_choice'][2:, 1] = out['y'][:-2]
    if 'prev_answer' in weights:
        out['inputs']['prev_answer'] = np.ones((len(out['answer']), 2))
        out['inputs']['prev_answer'][1:, 0] = out['answer'][:-1]
        out['inputs']['prev_answer'][2:, 1] = out['answer'][:-2]

    if 'prev_reward' in weights:
        out['inputs']['prev_reward'] = np.ones((len(reward), 2))
        out['inputs']['prev_reward'][1:, 0] += reward[:-1]
        out['inputs']['prev_reward'][2:, 1] += reward[:-2]
    if 'prev_punish' in weights:
        out['inputs']['prev_punish'] = np.ones((len(punish), 2))
        out['inputs']['prev_punish'][1:, 0] += punish[:-1]
        out['inputs']['prev_punish'][2:, 1] += punish[:-2]

    if 'cum_reward' in weights:
        out['inputs']['cum_reward'] = cum_reward
    if 'cum_punish' in weights:
        out['inputs']['cum_punish'] = cum_punish

    return out
