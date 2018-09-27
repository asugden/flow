from copy import deepcopy
import math
import numpy as np
import os.path as opath
from uuid import uuid1
import warnings

from . import config
from .misc import loadmat


class Trace2P:
    """
    Trace2P opens and manipulates simplified cellsort files. Pass the
    path to a simplified cellsort .mat file."""

    def __init__(self, path):
        self.d = loadmat(path)
        self._loadoffsets(path)

        # Instance variables
        self.ncells = np.shape(self.d['deconvolved'])[0]
        self.nframes = np.shape(self.d['deconvolved'])[1]
        self.framerate = 15.49 if 'framerate' not in self.d else self.d['framerate']
        if 'onsets' in self.d:
            self.trials = np.copy(self.d['onsets']).astype(np.int32)
        if 'offsets' in self.d:
            self.offsets = np.copy(self.d['offsets']).astype(np.int32)
        if 'conditions' in self.d:
            self.conditions = np.copy(self.d['condition'])
        self.type = 'training' if 'onsets' in self.d else 'spontaneous'

        # Set some optional variables
        self.cst = 2  # CS time
        self._addedcses = ['reward', 'punishment']

        # Clean things up
        self._loadextramasks(path)
        self._fixmistakes()
        self._initalize_roi_ids()

    @property
    def roi_ids(self):
        return self._roi_ids

    def ncs(self, cs):
        """Return the number of conditioned stimuli for condition cs."""
        onsets = self._onsets(cs)
        return len(onsets)

    def csonsets(self, cs='', errortrials=-1, lickcutoff=-1, lickwindow=(-1, 0)):
        """
        Return the trial onset times of a particular stimulus, separating by
        trial performance and licking

        :param cs: stimulus name, str. If blank, return all trials
        :param errortrials: whether to include all trials (-1), correct trials (0), or error trials (1)
        :param lickcutoff: cutoff number of licks allowed if >= 0
        :param lickwindow: lick window in which trials will be removed if lick number > lickcutoff
        :return: vector of frames of matching trial onsets
        """

        if cs == 'lick':
            return self.lickbout()

        if lickcutoff < 0:
            return self._onsets(cs, errortrials)
        else:
            ons = self._onsets(cs, errortrials)
            licks = self.licking()
            fr = (int(round(lickwindow[0] * self.framerate)),
                  int(round(lickwindow[1] * self.framerate)))
            out = []
            for o in np.array(ons):
                if np.sum(licks[o + fr[0]:o + fr[1]]) <= lickcutoff:
                    out.append(o)
            return out

    def csoffsets(self, cs='', errortrials=-1):
        """
        Return the stimulus offset times for a particular stimulus, separated by trial type

        :param cs: stimulus name, str. If blank, return all trials
        :param errortrials: whether to include all trials (-1), correct trials (0), or error trials (1)
        :return: vector of frames during which the stimulus turned off
        """

        if cs not in self.codes:
            return []

        out = np.copy(self.offsets)
        terr = np.copy(self.d['trialerror'])[:len(out)]
        cond = np.copy(self.d['condition'])[:len(out)]

        if len(cs) == 0 and errortrials > -1:
            out = out[terr%2 == errortrials]
        elif len(cs) > 0:
            out = out[cond == self.codes[cs]]

            if errortrials > -1:
                errs = terr[cond == self.codes[cs]]
                out = out[errs%2 == errortrials]

        return out

    def trialmask(self, cs='', errortrials=-1, fulltrial=False, padpre=0, padpost=0):
        """
        Return a mask which is true only for trials of type cs. Can return masks for stimuli or full trials.

        Parameters
        ----------
        cs : str
            Stimulus name, i.e. plus.
        errortrials : int
            All trials: -1, correct trials: 0, error trials: 1
        fulltrial : bool
            Mask between stimulus onsets if true, otherwise between stimulus onsets and offsets
        padpre : float
            Number of seconds to pad prior to stimulus onset
        padpost : float
            Number of seconds to pad post stimulus offset

        Returns
        -------
        ndarray
            Mask of length nframes in which True values mark the times of trials

        """

        out = np.zeros(self.nframes) > 1
        onsets = self._onsets(cs, errortrials)
        if len(onsets) == 0:
            return out

        padpre, padpost = int(round(padpre*self.framerate)), int(round(padpost*self.framerate))

        if fulltrial:
            trial_length = int(np.nanmedian(self.trials[1:] - self.trials[:-1]))
            all_offsets = np.append(self.trials, self.trials[-1] + trial_length) - 1
        else:
            if 'offsets' in self.d:
                tends = self.d['offsets'].flatten().astype(np.int32)
            else:
                print('WARNING: Estimating trial offsets')
                tends = onsets + int(round(2*self.framerate + 0.5))

            all_offsets = np.append(tends, min(tends[-1] + np.nanmedian(tends).astype(np.int32), self.nframes))

        for ons in onsets:
            offs = all_offsets[np.argmax(all_offsets > ons)]
            out[ons - padpre:offs + padpost] = 1

        return out

    def lastonset(self, pads=8):
        """
        Return the last stimulus onset, especially useful for days 9-11.
        This can be padded by pads seconds. Returns frame number
        """

        if 'onsets' not in self.d or len(self.d['onsets']) == 0:
            return 0
        else:
            padf = int(round(pads * self.framerate))
            return int(round(self.d['onsets'][-1])) + padf

    def cstraces(
            self, cs, start_s=-1, end_s=2, trace_type='deconvolved',
            cutoff_before_lick_ms=-1, errortrials=-1, baseline=None):
        """Return the onsets for a particular cs with flexibility.

        Parameters
        ----------
        cs : string
            CS-type to return traces of, Should be one of values returned bvy
            t2p.cses().
        start_s : float
            Time before stim to include, in seconds. For backward compatability, can also be arg dict.
        end_s : float
            Time after stim to include, in seconds.
        trace_type : {'deconvolved', 'raw', 'dff'}
            Type of trace to return.
        cutoff_before_lick_ms : int
            Exclude all time around licks by adding NaN's this many ms before
            the first lick after the stim.
        errortrials : {-1, 0, 1}
            -1 is all trials, 0 is only correct trials, 1 is error trials
        baseline : tuple of 2 ints, optional
            Use this interval (in seconds) as a baseline to subtract off from
            all traces each trial.

        Returns
        -------
        ndarray
            ncells x frames x nstimuli/onsets

        """
        defaults = {
            'start-s': -1 if isinstance(start_s, dict) else start_s,
            'end-s': end_s,
            'trace-type': trace_type,
            'cutoff-before-lick-ms': cutoff_before_lick_ms,
            'error-trials': errortrials,
            'baseline': baseline,
        }

        if isinstance(start_s, dict):
            args = deepcopy(start_s)
            start_s = defaults['start-s']
            for p in args:
                defaults[p] = args[p]

        defaults['start-fr'] = int(round(defaults['start-s']*self.framerate))
        defaults['end-fr'] = int(round(defaults['end-s']*self.framerate))
        defaults['cutoff-fr'] = int(round(defaults['cutoff-before-lick-ms']/
                                    1000.0*self.framerate))

        # Get lick times and onsets
        licks = self.d['licking'].flatten()
        ons = self.csonsets(cs, errortrials=defaults['error-trials'])
        out = np.empty((self.ncells, defaults['end-fr'] - defaults['start-fr'],
                        len(ons)))
        out.fill(np.nan)

        # Iterate through onsets, find the beginning and end, and add
        # the appropriate trace type to the output
        for i, onset in enumerate(ons):
            start = defaults['start-fr'] + onset
            end = defaults['end-fr'] + onset

            if i + start >= 0:
                if defaults['cutoff-before-lick-ms'] > -1:
                    postlicks = licks[licks > onset]
                    if len(postlicks) > 0 and postlicks[0] < end:
                        end = postlicks[0] - defaults['cutoff-fr']
                        if end < onset:
                            end = start - 1

                if end > self.nframes:
                    end = self.nframes
                if end > start:
                    out[:, :end-start, i] = self.trace(
                        defaults['trace-type'])[:, start:end]

        # Subtract the baseline, if desired
        if baseline is not None and \
                defaults['baseline'][0] != defaults['baseline'][1]:
            blargs = {key: defaults[key] for key in defaults}
            blargs['start-s'], blargs['end-s'] = blargs['baseline']
            blargs['baseline'] = (-1, -1)
            blargs['cutoff-before-first-lick-ms'] = -1
            bltrs = np.nanmean(self.cstraces(cs, blargs), axis=1)
            for f in range(np.shape(out)[1]):
                out[:, f, :] -= bltrs

        return out

    def outcomes(self, cs, errortrials=-1, maxdiff=6):
        """
        Return the outcome times for each trial, i.e. ensure or quinine

        :param cs: stimulus type str, plus, neutral or minus
        :param errortrials: -1 is all trials, 0 is correct trials, 1 is error trials
        :param maxdiff: the maximum difference in seconds between the stimulus and outcome.
        :return: array of presentations for each stimulus type
        """

        onsets = self.csonsets(cs, errortrials=errortrials)
        outcs = self.ensure() if cs == 'plus' or cs == 'pavlovian' else self.quinine() if cs == 'minus' else []
        outcs = np.array(outcs, dtype=np.int32)
        maxdiff = maxdiff*self.framerate

        out = []
        for ons in onsets:
            toutcs = outcs - ons
            nextout = toutcs[toutcs > 0]

            if len(nextout) > 0 and nextout[0] < maxdiff:
                out.append(nextout[0])
            else:
                out.append(-1)

        return np.array(out)

    def inversecstraces(self, cs, args):
        """
        Return the traces for trials in which there were not presentations of a particualr type.
        """

        # TODO: remove?

        # Set arguments and convert to framerate
        defaults = {
            'start-s': -1,
            'end-s': 2,
            'trace-type': 'deconvolved',
            'error-trials': -1,  # -1 is off, 0 is only correct trials, 1 is error trials
            'baseline': (-1, -1),  # Baseline removed if these two are different
        }

        for p in args:
            defaults[p] = args[p]

        if cs == 'ensure':
            cons = np.array(self.csonsets('plus', 0))
            fons = np.array(self.csonsets('plus', 1))
            eons = np.array(self.csonsets(cs, -1))
        else:  # elif cs == 'quinine':
            cons = np.array(self.csonsets('minus', 1))
            fons = np.array(self.csonsets('minus', 0))
            eons = np.array(self.csonsets(cs, -1))

        if len(eons) < 1:
            return []

        matches = []
        for co in cons:
            pos = np.argmax(eons > co)
            if eons[pos] > co and eons[pos] - co < self.framerate*4.1:
                matches.append(eons[pos] - co)

        if len(matches) < 1:
            return []

        off = int(np.round(np.nanmean(matches)))
        ons = fons + off

        defaults['start-fr'] = int(round(defaults['start-s']*self.framerate))
        defaults['end-fr'] = int(round(defaults['end-s']*self.framerate))

        # Get lick times and onsets
        out = np.zeros((self.ncells, defaults['end-fr'] - defaults['start-fr'], len(ons)))
        out[:, :, :] = np.nan

        # Iterate through onsets, find the beginning and end, and add
        # the appropirate trace type to the output
        for i, onset in enumerate(ons):
            start = defaults['start-fr'] + onset
            end = defaults['end-fr'] + onset

            if end > self.nframes:
                end = self.nframes
            if end > start:
                out[:, :end-start, i] = self.trace(defaults['trace-type'])[:, start:end]

        # Subtract the baseline, if desired
        if defaults['baseline'][0] != defaults['baseline'][1]:
            blargs = {key: defaults[key] for key in defaults}
            blargs['start-s'], blargs['end-s'] = blargs['baseline']
            blargs['baseline'] = (-1, -1)
            blargs['cutoff-before-first-lick-ms'] = -1
            bltrs = np.nanmean(self.cstraces(cs, blargs), axis=1)
            for f in range(np.shape(out)[1]):
                out[:, f, :] -= bltrs

        return out

    def firstlick(self, cs, units='frames', errortrials=-1, maxframes=-1):
        """
        Return the first lick time for all onsets of type cs
        NOTE: Assumes trial structure of 6 second ITI

        :param cs: stimulus type str: plus, minus, or neutral
        :param units: return the output in frames, seconds (s) or milliseconds (ms)
        :param errortrials: -1 is all trials, 0 is correct trials, 1 is error trials
        :param maxframes: the latest time acceptable in frames
        :return:
        """

        ons = self.csonsets(cs, errortrials=errortrials)
        licks = self.d['licking'].flatten()
        ntrialframes = int(round(8*self.framerate))
        if maxframes < 0:
            maxframes = ntrialframes
        out = []

        # Iterate through onsets, find the beginning and end, and add
        # the appropirate trace type to the output
        for onset in ons:
            postlicks = licks[licks > onset]
            if len(postlicks) > 0:
                out.append(postlicks[0] - onset)
            else:
                out.append(np.nan)

        # Convert to float so that nans can be added
        out = np.array(out, dtype=np.float32)
        with warnings.catch_warnings():
            # We're ignoring warnings of comparisons between nans already in the array and nan
            warnings.simplefilter('ignore', category=RuntimeWarning)
            nnan = np.isfinite(out)
            out[nnan][out[nnan] > maxframes] = np.nan

        if units[0] == 's':
            out /= self.framerate
        elif units[0:2] == 'ms':
            out = 1000*out/self.framerate

        return np.array(out)

    def stimlicks(self, cs='', mins=2, maxs=4, errortrials=-1):
        """
        Return the first lick time for all onsets of type cs
        NOTE: Assumes trial structure of 6 second ITI

        :param cs: stimulus type str: plus, minus, or neutral, or empty string for all types
        :param mins: min time in seconds
        :param maxs: max time in seconds
        :param errortrials: -1 is all trials, 0 is correct trials, 1 is error trials
        :return: counts of licking in the time interval for each trial of type cs
        """

        # TODO: combine w/ firstlick?

        ons = self.csonsets(cs, errortrials=errortrials)
        licks = self.d['licking'].flatten()
        ntrialframes = int(round(8*self.framerate))
        out = []

        minf = int(round(mins*self.framerate))
        maxf = int(round(maxs*self.framerate))

        # Iterate through onsets, find the beginning and end, and add
        # the appropirate trace type to the output
        for onset in ons:
            postlicks = licks[(licks >= onset + minf) & (licks < onset + min(maxf, ntrialframes))]
            out.append(len(postlicks))

        return np.array(out)

    def cses(self, ensurequinine=False):
        """
        Return all of the shown stimuli, and include ensure and quinine presentations if they exist.

        :param ensurequinine: if true, add ensure and quinine presentations if they exist in the file.
        :return: list of strings of types of stimulus presentations
        """

        out = [key for key in self.codes] + [key for key in self._addedcses]
        if ensurequinine:
            if 'ensure' in self.d:
                out.append('ensure')
            if 'quinine' in self.d:
                out.append('quinine')

        return out

    def trace(self, tracetype):
        """
        Return trace of type 'axondff' or 'raw'. ncell < 0 returns all.
        Returned in form [cell, time]
        """

        # Types usually are dff, raw, f0, dec/deconvolved
        if tracetype.lower()[:3] == 'dec':
            tracetype = 'deconvolved'
        return self.d[tracetype]

    def pupilmask(self, include_phase=False):
        """
        Return the mask by pupil diameter and if desired, phase.
        True outputs are times that should be KEPT, false should be removed
        """

        # They other masks are not important, active and passive are estimates
        # pupil_mask is based on the actual number of pixels,
        # calibrated relative to the eyeball diameter

        # TODO : remove
        if 'pupil_mask' not in self.d:
            if 'pupil' in self.d:
                # print('WARNING: Mask not included in trace2p file.')
                return np.ones(self.nframes) > 0
                # exit(0)
            else:
                return np.ones(self.nframes) > 0

        out = self.d['pupil_mask'] > 0
        out = out.flatten()[:self.nframes]

        if len(out) < self.nframes:
            tempout = np.zeros(self.nframes) < 1
            tempout[:len(out)] = out
            out = tempout

        # print 'Remaining percent after masking: %.2f' % (100*float(np.sum(self.d['pixel_mask']))/self.nframes)
        if False and include_phase:
            out[self.d['deriv_mask'].flatten()[:self.nframes] < 1] = 0
        return out

    def lickbout(self, ili=2):
        """
        Return the time of lickbout onsets

        :param ili: inter-lick interval in seconds
        :return: vector of lickbout onset times
        """

        lickbout = np.zeros(self.nframes)
        lickbout[self.licking()] = 1
        conv = np.zeros(int(round(ili*self.framerate)) - 1)
        conv[int(round(len(conv)/2)):] = 1
        lickbout = np.convolve(lickbout, conv, 'same')
        lickbout[lickbout > 0] = 1
        lickbout = lickbout[1:] - lickbout[:-1]

        return np.arange(len(lickbout))[lickbout > 0]

    def inactivity(self, nostim=True, runsec=10, motsec=3, licksec=10, run_min=4, mot_stdev=4):
        """
        Return time periods of inactivity as defined by brain motion and running.

        :param nostim: eliminate all times of stimuli if True
        :param runsec: number of seconds to expand times of running
        :param motsec: number of seconds to expand times of brain motion
        :param licksec: number of seconds to expand times around licking
        :param run_min: minimum centimeters per second defining running
        :param mot_stdev: standard deviation for active brain motion periods
        :return: boolean vector where true represents inactivity
        """

        stm = np.zeros(self.nframes) > 1

        if nostim:
            stm[:self.lastonset()] = True

        runf = np.zeros(self.nframes)
        run = self.speed()[:self.nframes] > run_min
        if isinstance(run, bool):
            print('WARNING: Running not recorded')
            run = runf
        elif len(run) < len(runf):
            print('WARNING: Run lengths do not match')
            runf[-1*len(run):] = run
            run = runf
        run = np.convolve(run, np.ones(int(round(runsec*self.framerate)), dtype=np.float32), mode='same') > 0

        mot = self.motion(True)[:self.nframes]
        mot = mot - np.nanmean(mot)
        mot = np.abs(mot) > mot_stdev*np.nanstd(mot)
        mot = np.convolve(mot, np.ones(int(round(motsec*self.framerate)), dtype=np.float32), mode='same') > 0

        lck = np.zeros(self.nframes)
        if len(self.licking()) > 0:
            lck[self.licking()] = 1
            lck = np.convolve(lck, np.ones(int(round(licksec*self.framerate)), dtype=np.float32), mode='same')
        lck = lck > 0

        return np.invert(np.bitwise_or(run, np.bitwise_or(mot, np.bitwise_or(lck, stm))))

    def hasvar(self, var):
        """
        Check if trace2p has variable type

        :param var: string: photometry, pupil, pupilmask, ripple
        :return: boolean
        """

        if var == 'pupil':
            return self.haspupil()
        elif var == 'photometry':
            return self.hasphotometry()
        elif var == 'pupilmask':
            return self.haspupilmask()
        elif var == 'ripple':
            return self.hasripple()
        elif var == 'running':
            if 'running' not in self.d or len(self.d['running'].flatten()) == 0:
                return False
            else:
                return True
        elif var == 'brainmotion':
            if 'brainmotion' not in self.d or len(self.d['brainmotion'].flatten()) == 0:
                return False
            else:
                return True
        else:
            return False

    def getvar(self, var, tracetype=''):
        """
        Return a variable such as pupil, photometry, etc.

        :param var: string: photometry, pupil, pupilmask, ripple
        :param tracetype: optional trace type for photometry or ripple
        :return: boolean
        """

        if var == 'pupil':
            return self.pupil()
        elif var == 'photometry':
            return self.photometry(tracetype=tracetype)
        elif var == 'ripple':
            return self.ripple()
        elif var == 'running':
            return self.speed()
        elif var == 'brainmotion':
            return self.motion(True)
        else:
            return None

    def haspupilmask(self):
        """
        Return whether a pupil mask exists (without setting all values to 1

        :return: boolean
        """

        if ('pupil' in self.d
                and 'pupil_mask' in self.d
                and len(self.d['pupil'].flatten()) > 0
                and len(self.d['pupil_mask'].flatten()) > 0):
            return True
        else:
            return False

    def hasphotometry(self):
        """
        Return whether the trace file has photometry

        :return: boolean
        """

        if 'photometry' in self.d and len(self.d['photometry'].flatten()) > 0:
            return True

    def hasripple(self):
        """
        Return whether the trace file has hippocampal ripple data

        :return: boolean
        """

        if 'ripple' in self.d and len(self.d['ripple'].flatten()) > 0:
            return True
        return False

    def haspupil(self):
        """
        Return whether the trace file has pupil data

        :return:
        """

        if 'pupil' not in self.d or len(self.d['pupil']) == 0:
            return False
        else:
            return True

    def hasoffsets(self):
        """Return whether the offsets variable exists.

        :return: boolean
        """

        if 'offsets' not in self.d or len(self.d['offsets']) == 0:
            return False
        else:
            return True

    def nocs(self, length, safety_frames, running_threshold=-1):
        """
        Get a series of time points of length length with a buffer of
        safety_frames away from any stimulus that comprise all times
        during which stimuli are not being shown.
        """

        # Get a list of all stimulus onsets
        cses = []
        for code in self._addedcses:
            cses.extend(self._onsets(code))
        for code in self.codes:
            cses.extend(self._onsets(code))
        cses.sort()

        # Allow safety_frames to be a tuple or an integer
        if isinstance(safety_frames, int):
            safety_frames = (safety_frames, safety_frames)

        # Set the time period of the safety-padded start and end times
        stimlen = int(math.ceil(self.cst*self.framerate))
        starts = np.array([0] + cses) + stimlen + safety_frames[0]
        ends = np.array(cses + [self.nframes]) - safety_frames[1]

        # Add runnning speed cutoff
        starts, ends = self._limit_to_running(starts, ends, running_threshold)

        # Calculate all chunks that fit within those times
        out = []
        for i in range(len(starts)):
            j = starts[i]
            while j + length <= ends[i]:
                out.append(j)
                j += length
        return out

    def speed(self):
        """
        Return a vector of running speed if possible. Speed is in cm/s smoothed over a second.
        """

        wheel_diameter = 14  # in cm
        wheel_tabs = 44
        wheel_circumference = wheel_diameter*math.pi
        step_size = wheel_circumference/(wheel_tabs*2)

        # Added
        if 'running' not in self.d or len(self.d['running'].flatten()) == 0:
            return []

        if (len(np.shape(self.d['running'])) > 1
                and np.shape(self.d['running'])[0] == 1
                and np.shape(self.d['running'])[1] > 1):
            self.d['running'] = self.d['running'].flatten()

        instantaneous_speed = np.zeros(len(self.d['running']))
        if len(instantaneous_speed) > 0:
            instantaneous_speed[1:] = self.d['running'][1:] - self.d['running'][:-1]
            instantaneous_speed[1] = 0
            instantaneous_speed = instantaneous_speed.flatten()*step_size*self.framerate

            intfr = int(math.ceil(self.framerate))
            speed = np.convolve(instantaneous_speed, np.ones(intfr, dtype=np.float32)/intfr, mode='same')

            return speed
        else:
            print('Running not included')
            return np.zeros(self.nframes)

    def behavior(self, cs, aspercent=True):
        """
        Return the success of the behavior for a particular stimulus.
        If aspercent is set to false, returns correct, total.
        """

        conds = [self.d['condition'].flatten() == self.codes[cs]]
        num = np.sum(self.d['trialerror'][conds]%2 == 0)
        denom = np.sum(conds)

        if aspercent:
            return float(num)/denom
        return num, denom

    def errors(self, cs):
        """
        Return the error codes of trials for a particular cs.

        :param cs: plus, minus, neutral, blank, pavlovian
        :return: vector of error codes with 1s being errors
        """

        if cs not in self.codes:
            return []

        lnonsets = len(self.d['onsets'])
        conds = [self.d['condition'].flatten()[:lnonsets] == self.codes[cs]]
        minlen = min(np.shape(conds)[1], np.shape(self.d['onsets'])[0], np.shape(self.d['trialerror'])[0])
        conds = conds[:minlen]
        out = self.d['trialerror'].flatten()[:minlen][tuple(conds)] % 2
        out = [o for o in out.flatten() if o < self.nframes]

        return np.array(out) == 1

    def licking(self):
        """Return the frames in which there was a lick onset."""

        if isinstance(self.d['licking'], int):
            self.d['licking'] = np.array([self.d['licking']])

        licks = self.d['licking'].flatten()
        licks = licks[licks < self.nframes]
        return licks

    def ensure(self):
        """Return the frames in which there was ensure delivered."""
        if 'onsets' not in self.d:
            return np.array([])
        return self.d['ensure'].flatten()

    def quinine(self):
        """Return the frames in which there was quinine delivered."""
        if 'onsets' not in self.d:
            return np.array([])
        return self.d['quinine'].flatten()

    def motion(self, diff=False):
        """
        Return the distance traveled by the brain as judged by
        registration.

        :param diff: Return the mean-subtracted diff of motion if true
        :returns: vector of brainmotion of length nframes
        """

        if 'brainmotion' not in self.d:
            raise ValueError('brain motion not included in simpcell file')

        if diff:
            bm = self.d['brainmotion'].flatten()[:self.nframes]
            out = np.zeros(self.nframes)
            out[1:] = bm[1:] - bm[:-1]
            out -= np.nanmean(out)
            return out
        else:
            return self.d['brainmotion'].flatten()

    def pupil(self):
        """
        Return the pupil diameter over time.
        """

        if 'pupil' not in self.d or len(self.d['pupil']) == 0:
            return []

        pup = np.copy(self.d['pupil'].flatten()[:self.nframes])
        if len(pup) < self.nframes:
            print('WARNING: Not enough frames in pupil')
            out = np.zeros(self.nframes)
            out[:] = np.max(pup)
            out[:len(pup)] = pup
            pup = out
        return pup

    def photometry(self, fiber=0, tracetype='dff'):
        """
        Return the photometry vector

        :return: numpy vector or empty list
        """

        if 'photometry' in self.d and len(self.d['photometry'].flatten()) > 0:
            if tracetype == 'dff':
                if self.d['photometry'].ndim == 1 and fiber == 0:
                    return self.d['photometry']
                elif self.d['photometry'].ndim == 2:
                    return self.d['photometry'][fiber, :]
                else:
                    return []
            elif tracetype == 'raw':
                return self.d['photometryraw'].flatten()
            else:
                if self.d['photometrydeconvolved'].ndim == 1 and fiber == 0:
                    return self.d['photometrydeconvolved']
                elif self.d['photometrydeconvolved'].ndim == 2:
                    return self.d['photometrydeconvolved'][fiber, :]
                else:
                    return []
        else:
            return []

    def _limit_to_running(self, ons, offs, runmin, runmax=9999, speeds=None, printout=False):
        """
        Given a series of onsets and offsets, reset them so that they
        only include running speeds between a certain range. May not be
        able to handle skipping frames. Spread accepts times before the
        running threshold is reached. This account for premotor activity
        """

        if runmin < 0 and runmax == 9999:
            return ons, offs

        run = self.speed() if speeds is None else speeds
        rtimes = np.zeros(len(run), dtype=bool)
        rtimes[run >= runmin] = 1
        rtimes[run > runmax] = 0

        if printout:
            print('Limiting to  %.1f < speed < %.1f cm/s limits to %.1f%% of movie\n' % (
                runmin, runmax, float(np.sum(rtimes)) / len(run) * 100))

        otimes = np.zeros(len(run), dtype=bool)
        for i in range(len(ons)):
            otimes[ons[i]:offs[i]] = 1

        times = np.bitwise_and(rtimes, otimes).astype(np.int8)
        dtimes = np.zeros(len(run))
        dtimes[1:] = times[1:] - times[:-1]

        ons = np.arange(len(run))[dtimes > 0]
        offs = np.arange(len(run))[dtimes < 0]
        return ons, offs

    def centroids(self):
        """Return the centroids of all neurons.

        """
        # medial-top, anterior-right (for all mice after CB173)
        # so, higher X is anterior, higher Y is lateral
        # let's convert to higher X is lateral, higher Y is posterior

        lateralness = self.d['centroid'][:, 1]/512.0 - 0.5
        posteriorness = 0.5 - self.d['centroid'][:, 0]/796.0

        # return self.d['centroid']
        return lateralness, posteriorness

    def conderrs(self):
        """Get a list of all onsets, errors, and codes.

        :return: onsets, errors, codes

        """

        # TODO: remove?

        maxlen = len(self.d['onsets'].flatten())
        cond = self.d['condition'].flatten()[:maxlen]
        errs = self.d['trialerror'].flatten()[:maxlen] % 2

        return cond, errs, self.codes

    def _onsets(self, cs='', errortrials=-1):
        """
        Return the onset frames of stimulus cs

        :param cs: stimulus name, including reward, punishment, quinine, or blank for all trials
        :param errortrials: -1 is all trials, 0 is correct trials, 1 is error trials
        :return: vector of stimulus onset frames
        """

        if 'onsets' not in self.d:
            return []

        # Account for all trial types
        if len(cs) == 0:
            out = np.copy(self.d['onsets'])
            if errortrials > -1:
                out = out[self.d['trialerror']%2 == errortrials]
        elif cs == 'reward' or cs == 'ensure':
            out = np.copy(self.d['ensure'])
        elif cs == 'punishment' or cs == 'quinine':
            out = np.copy(self.d['ensure'])
        else:
            if cs not in self.codes:
                return []
            out = np.copy(self.d['onsets'])
            out = out[self.d['condition'] == self.codes[cs]]
            if cs != 'pavlovian' and errortrials > -1:
                errs = np.copy(self.d['trialerror'])[self.d['condition'] == self.codes[cs]]
                out = out[errs%2 == errortrials]

        out = out[(out > 0) & (out < self.nframes)]

        return out.astype(np.int32)

    def _loadextramasks(self, path):
        """Load an extra mask file if necessary."""
        # TODO: remove
        mpath = path.replace('.simpcell', '-pupilrepmask.mat')
        if opath.exists(mpath):
            masks = loadmat(mpath)
            for key in masks:
                self.d[key] = masks[key]

    def _loadoffsets(self, path):
        """Load an extra mask file if necessary."""
        # TODO: remove
        params = config.params()
        datad = params['paths'].get('data', '/data')
        mpath = opath.split(path)[1].replace('.simpcell', '.onsets')
        path = opath.join(datad, 'onsets/%s' % mpath)

        if opath.exists(path):
            offsets = loadmat(path)
            self.d['offsets'] = offsets['offsets']

    def _fixmistakes(self):
        """
        Fix errors in the matlab file such as more trial errors than trial onsets, due to
        not including those onsets that go beyond the end of the file

        :return:
        """

        self.codes = self.d['codes'] if 'codes' in self.d else {}

        if 'trialerror' in self.d and len(self.d['trialerror']) > len(self.d['onsets']):
            self.d['trialerror'] = self.d['trialerror'][:len(self.d['onsets'])]
        if 'condition' in self.d and len(self.d['condition']) > len(self.d['onsets']):
            self.d['condition'] = self.d['condition'][:len(self.d['onsets'])]

    def _initalize_roi_ids(self):
        """Give each ROI a unique ID.

        This clearly needs to pull this info from somewhere else if it already
        exists, but I just don't know what that looks like.
        For now just give each ROI and ID that should be unique.

        """
        self._roi_ids = tuple(str(uuid1()) for _ in range(self.ncells))

# Version implemented in flow.misc
# def loadmatpy(filename):
#     """
#     A modified loadmat that can account for structs as dicts.
#     """

#     data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True, appendmat=False)
#     for key in data:
#         if isinstance(data[key], spio.matlab.mio5_params.mat_struct):
#             data[key] = _mattodict(data[key])
#     return data

# def _mattodict(matobj):
#     """
#     Recursively convert matobjs into dicts.

#     :param matobj: matlab object from _check_keys
#     :return: dict
#     """

#     out = {}
#     for strg in matobj._fieldnames:
#         el = matobj.__dict__[strg]
#         if isinstance(el, spio.matlab.mio5_params.mat_struct):
#             out[strg] = _mattodict(el)
#         else:
#             out[strg] = el
#     return out
