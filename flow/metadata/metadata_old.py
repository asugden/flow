from __future__ import print_function
from builtins import str
from copy import deepcopy
import os
import pandas as pd

from . import metadata, parser

# This is used to cache the metadata parsed as a dataframe.
_dataframe = None


"""
training, running always hungry
sated-stim always sated

dates are spontaneous, hungry/sated based on parsed group

disengaged, naive as date tags

skip-crossday added as date tag

add replay1 mouse tag to:
CB173
AS20
OA32
OA34
OA36
OA37
OA38
AS41

"""

def parse_spontaneous(overwrite=False):
    if overwrite:
        os.remove(parser._get_metadata_path())
    replay1_mice = ['CB173', 'AS20', 'OA32', 'OA34', 'OA36', 'OA37', 'OA38', 'AS41']
    jeff_mice = ['OA178', 'OA191', 'OA192']
    for mouse in spontaneous:
        if mouse in replay1_mice:
            mouse_tags = ['replay1']
        elif mouse in jeff_mice:
            mouse_tags = ['jeff']
        else:
            mouse_tags = []
        print("Adding {}".format(mouse))
        metadata.add_mouse(mouse, tags=mouse_tags)
        for group in spontaneous[mouse]:
            date_tags = []
            sated = 'sated' in group
            hungry = 'hungry' in group
            if 'disengaged' in group:
                date_tags.append('disengaged')
            if 'naive' in group:
                date_tags.append('naive')
            if spontaneous[mouse][group].get('skip-crossday', False):
                date_tags.append('skip-crossday')
            training_runs = spontaneous[mouse][group]['train']
            running_runs = spontaneous[mouse][group]['running']
            sated_stim_runs = spontaneous[mouse][group].get('sated-stim', [])
            photometry = spontaneous[mouse][group].get('photometry', [])
            for key in spontaneous[mouse][group]:
                if key in ('train', 'running', 'sated-stim', 'photometry',
                           'skip-crossday'):
                    continue
                date = int(key)
                print('Adding {}-{}'.format(mouse, date))
                try:
                    metadata.add_date(
                        mouse, date, photometry=photometry, tags=date_tags)
                except metadata.AlreadyPresentError:
                    pass

                for run in training_runs:
                    run_tags = ['hungry']
                    try:
                        metadata.add_run(
                            mouse, date, run, run_type='training', tags=run_tags)
                    except metadata.AlreadyPresentError:
                        pass

                for run in running_runs:
                    run_tags = ['hungry']
                    try:
                        metadata.add_run(
                            mouse, date, run, run_type='running', tags=run_tags)
                    except metadata.AlreadyPresentError:
                        pass

                for run in sated_stim_runs:
                    run_tags = ['sated']
                    try:
                        metadata.add_run(
                            mouse, date, run, run_type='sated-stim', tags=run_tags)
                    except metadata.AlreadyPresentError:
                        pass

                for run in spontaneous[mouse][group][key]:
                    assert (sated or hungry) and not (sated and hungry)
                    if sated:
                        run_tags = ['sated']
                    elif hungry:
                        run_tags = ['hungry']
                    else:
                        run_tags = []
                    metadata.add_run(
                        mouse, date, run, run_type='spontaneous', tags=run_tags)

    return metadata.meta(sort=True)


spontaneous = {
    'CB173': {
        'group-6-8-sated': {
            'train': [2, 3, 4],
            'running': [1, 5],
            'sated-stim': [7],
            '160503': [8],
            '160506': [6, 8],
            '160513': [6, 8],
            '160516': [6, 8],
            '160517': [6, 8],
            '160519': [6, 8],
            '160520': [6, 8],
            '160523': [6, 8],
        },

        'group-6-sated-no-stim': {
            'train': [2, 3, 4],
            'running': [1, 5],
            '160509': [6],
        },

        'group-missed-5-sated': {
            'train': [2, 3, 4],
            'running': [1],
            'sated-stim': [7],
            '160526': [6, 8],
        },

        'group-missed-2-sated': {
            'train': [3, 4],
            'running': [1, 5],
            'sated-stim': [7],
            '160527': [6, 8],
        }
    },

    'AS20': {
        'group-6-8-sated': {
            'train': [2, 3, 4],
            'running': [1, 5],
            'sated-stim': [7],
            '160816': [6, 8],
            '160818': [6, 8],
            '160819': [6, 8],
            '160825': [6, 8],
            '160826': [6, 8],
            '160827': [6, 8],
            '160829': [6, 8],  # massive brain motion
            '160830': [6, 8],  # massive brain motion
            '160907': [6, 8],
            '160908': [6, 8],
            '160915': [6, 8],
        },

        'group-6-8-sated-missed-7': {
            'train': [2, 3, 4],
            'running': [1, 5],
            '160822': [6, 8],
            '160919': [6, 8],
        },

        'group-9-11-sated': {
            'train': [2, 3, 4],
            'running': [1],
            '160920': [9, 10, 11],
            '160922': [9, 10, 11],
            '161027': [9, 10],  # CHECK, error with length of results
            '161028': [9, 10, 11],  # CHECK, error with length of results
            '161031': [9, 10, 11],  # CHECK, error with length of results
            # '161103': [9, 10, 11],  # Unusually high population activity, too much running,  and problems with
            # deconvolution
            '161107': [9, 10, 11],  # CHECK, error with length of results
        },

        'group-5-9-11-hungry': {
            'train': [2, 3, 4],
            'running': [1],
            '161027': [5],  # CHECK, error with length of results
            '161028': [5],  # CHECK, error with length of results
            '161031': [5],  # CHECK, error with length of results
            # '161103': [5],
            '161107': [5],  # CHECK, error with length of results
        },

        'group-missed-2-sated': {
            'train': [3, 4],
            'running': [1, 5],
            'sated-stim': [7],
            '160823': [6, 8],
        },

        'group-extra-sated-5-sated': {
            'train': [2, 3, 4],
            'running': [1],
            'sated-stim': [7],
            # '160906': [5, 6, 8],  # removed because of seg fault
        },
    },

    'AS21': {
        'group-9-11-sated': {
            'train': [2, 3, 4],
            'running': [1],
            '161017': [9, 10, 11],
            '161021': [9, 10, 11],
            '161024': [9, 10, 11],
            '161025': [9, 10, 11],
            '161101': [9, 10, 11],
            '161102': [9, 10, 11],  # Weird bug. Check
            '161108': [9, 10, 11],
            '161109': [9, 10, 11],
            '161115': [9, 10, 11],
        },

        'group-5-9-11-hungry': {
            'train': [2, 3, 4],
            'running': [1],
            '161101': [5],
            '161102': [5],
            '161108': [5],
            '161109': [5],
            '161115': [5],
        },

        'group-extralong-2-sated': {
            'train': [2, 4],
            'running': [1],
            '161018': [9, 10, 11],
        }
    },

    'AS23': {  # Seizures. Remove.
        'group-9-11-sated': {
            'train': [2, 3, 4],
            'running': [1],
            '161114': [9, 10, 11],
            '161116': [9, 10, 11],
            '161118': [9, 10],
            '161122': [9, 10, 11],
            '161128': [9, 10, 11],
            '161129': [9, 10, 11],
            '161202': [9, 10, 11],
            '161205': [9, 10, 11],
            '161206': [9, 10, 11],
            '161208': [9, 10, 11],
            '161212': [9, 10, 11],  # had problems with randomization for this day
            '161215': [9, 10, 11],
            '161216': [9, 10, 11],
            '161219': [9, 10, 11],
            '161230': [9, 10, 11],
            '170103': [9, 10, 11],
            '170105': [10, 11],
            '170113': [9, 10, 11],
            '170117': [9, 10, 11],
            '170123': [9, 10, 11],
        },

        'group-5-hungry': {
            'train': [2, 3, 4],
            'running': [1],
            '161114': [5],
            '161116': [5],
            '161122': [5],
            '161128': [5],
            '161129': [5],
            '161202': [5],
            '161205': [5],
            '161206': [5],
            '161208': [5],
            '161212': [5],  # had problems with randomization for this day
            '161215': [5],
            '161216': [5],
            '161219': [5],
            '161230': [5],
            '170103': [5],
            '170105': [5],
            '170113': [5],
            '170117': [5],
            '170123': [5],
        },

        # 'group-9-11-sated-overnight': {
        # 	'train': [2, 3, 4],
        # 	'running': [1],
        # 	'170127': [9, 10, 11],
        # },
        #
        # 'group-5-sated-overnight': {
        # 	'train': [2, 3, 4],
        # 	'running': [1],
        # 	'170127': [5],
        # },

        'group-9-11-sated-catch-trials': {
            'skip-crossday': True,
            'train': [2, 3, 4],
            'running': [1],
            '170131': [9, 10, 11],
        },

        'group-5-hungry-catch-trials': {
            'skip-crossday': True,
            'train': [2, 3, 4],
            'running': [1],
            '170131': [5],
        },
    },

    'OA32': {
        # 170421 missed 4 and photometry-- removed
        'group-9-11-sated-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            # '170324': [9, 10],  # Weird semi-rhythmic activity-- removing
            '170327': [9, 10, 11],  # High activity
            '170328': [9, 10, 11],
            '170330': [9, 10, 11],
            '170403': [9, 10, 11],
            '170406': [9, 10, 11],
            '170407': [9, 10, 11],
            '170409': [9, 10, 11],  # run 1 is not great, but works
            '170410': [9, 10, 11],
            '170412': [9, 10, 11],
            '170414': [10, 11],
            '170417': [9, 10, 11],
            '170420': [9, 10, 11],
        },

        'group-9-11-sated-ca1': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['ca1'],
            '170331': [9, 10, 11],
            '170404': [9, 10, 11],
        },

        'group-5-hungry-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '170328': [5],
            '170330': [5],
            # '170331': [5],
            '170403': [5],
            # '170404': [5],
            '170406': [5],
            '170407': [5],
            '170410': [5],
            '170412': [5],
            '170414': [5],
            '170417': [5],
            '170420': [5],
        },

        'group-5-hungry-ca1': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc', 'ca1'],
            '170331': [5],
            '170404': [5],
        },

        'group-9-11-sated-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '170424': [9, 10],
            '170426': [9, 10, 11],
            '170427': [5, 9, 10, 11],
            '170430': [9, 10, 11],
        },

        'group-5-hungry-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '170424': [5],
            '170426': [5],
            '170430': [5],
        },
    },

    'OA32-H': {
        'group-9-11-sated-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            # '170507': [9, 10, 11],  # Running was not saved-- cannot use
            '170508': [9, 10, 11],
            '170510': [9, 10, 11],
        },

        'group-5-hungry-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            # '170508': [5],  # No running added. We might be deciding only to use sated
            # '170510': [5],
        },

        'group-9-11-sated-no-phot-no-2': {
            'train': [3, 4],
            'running': [1],
            '170511': [9, 10, 11],
        },

        'group-5-hungry-no-phot-no-2': {
            'train': [3, 4],
            'running': [1],
            '170511': [5],
        },
    },

    'OA34': {
        'group-9-11-sated-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '170501': [9, 10, 11],
            '170502': [9, 10, 11],
            '170504': [9, 10, 11],
            '170505': [9, 10, 11],
            '170512': [9, 10, 11],
            '170515': [9, 10, 11],
            '170516': [9, 10, 11],
            '170518': [9, 10, 11],
            '170519': [9, 10, 11],
            '170522': [9, 10, 11],
            '170523': [9, 10, 11],
            '170525': [9, 10, 11],
            '170526': [9, 10, 11],
            '170531': [9, 10, 11],
            '170602': [9, 10, 11],
            '170605': [9, 10, 11],
            '170608': [9, 10, 11],
            '170612': [9, 10, 11],
            '170613': [9, 10, 11],  # No running in run 1, acceptable running through 2-4
            # '170615': [9, 10, 11],  # No running in run 1, even including other runs, only 3 running events
            '170616': [9, 10, 11],
            '170619': [9, 10, 11],
            '170622': [9, 10],
            '170623': [9, 10, 11],
            '170626': [9, 10, 11],
        },

        'group-5-hungry-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '170501': [5],
            '170502': [5],
            '170504': [5],
            '170505': [5],
            '170512': [5],
            '170515': [5],
            '170516': [5],
            '170518': [5],
            '170519': [5],
            '170522': [5],
            '170523': [5],
            '170525': [5],
            '170531': [5],
            '170602': [5],
            '170605': [5],
            '170608': [5],
            '170612': [5],
            '170613': [5],  # No running in run 1, acceptable running through 2-4
            # '170615': [5],  # No running in run 1, even including other runs, only 3 running events
            '170616': [5],
            '170619': [5],
            '170622': [5],
            '170623': [5],
            '170626': [5],
        },
    },

    'CB210-naive': {
        'naive-sated': {
            'train': [2, 3, 4],
            'running': [1],
            '170619': [9, 10, 11],
            '170620': [9, 10, 11],
        },
    },

    'OA34-dis': {
        'disengaged-sated': {
            'train': [2, 3, 4, 5],
            'running': [1],
            '170627': [9, 10, 11],
            '170629': [9, 10, 11],
            '170705': [9, 10, 11],
            '170706': [9, 10, 11],
            '170707': [9, 10, 11],
            '170710': [9, 10, 11],
        },
    },

    'OA35': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            '170815': [9, 10, 11],
        },

        'sated-mistrained': {
            'train': [2, 4],
            'running': [1],
            '170814': [9, 10, 11, 12],
        },
    },

    'OA36': {
        'sated-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            # '170727': [9, 11],  # Stress and water leak
            '170728': [9, 10, 11],
            '170801': [9, 10, 11],
            '170803': [9, 10, 11],
            '170804': [9, 10, 11],
            '170808': [9, 10, 11],
            '170817': [9, 10],
            '170818': [9, 10, 11],
            '170825': [9, 10, 11],
            '170828': [9, 10, 11],
            '170829': [9, 10, 11],
            '170831': [9, 10, 11],
            '170901': [9, 10, 11],
            '170905': [9, 10, 11],
            '170906': [9, 10, 11],
            '170908': [9, 10, 11],
            '170911': [9, 10, 11],
            # '170919': [9, 10, 11],  new field, do not use
        },

        'hungry-nacc': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            # '170727': [5],  # Stress and water leak
            '170728': [5],
            '170801': [5],
            '170803': [5],
            '170804': [5],
            '170817': [5],
            '170818': [5],
            '170825': [5],
            '170828': [5],
            '170829': [5],
            '170831': [5],
            '170901': [5],
            '170905': [5],
            '170908': [5],
            '170911': [5],
            # '170919': [5],  new field, do not use
        },
    },

    'OA36-H': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '171003': [9, 10, 11],
            '171004': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '171003': [5],
            '171004': [5],
        },
    },

    'OA37': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '171017': [9, 10, 11],
            '171020': [9, 10, 11],
            '171024': [9, 10, 11],
            '171029': [9, 10],
            '171030': [9],
            '171103': [9, 10, 11],
            '171106': [9, 10],
            '171115': [9, 10],
        },

        'hungry': {
            'train': [2, 3, 4],
            'photometry': ['nacc'],
            'running': [1],
            '171017': [5],
            '171020': [5],
            '171024': [5],
            '171030': [5],
            '171103': [5],
            '171106': [5],
        },

        'sated-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '171019': [9, 10, 11],
        },

        'hungry-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '171019': [5],
        },
    },

    'OA37-H': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '171005': [9, 10, 11],
            '171009': [9, 10],  # Run 11 has wrong number of ROIs and icamasks are missing
            '171010': [9, 10, 11],
            '171012': [9, 10, 11],
            '171013': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'photometry': ['nacc'],
            'running': [1],
            '171009': [5],
            '171010': [5],
        },
    },

    'OA38-naive': {
        'naive-sated': {
            'train': [2, 3, 4],
            'running': [1],
            '170922': [9, 10, 11],
            '170925': [9, 10, 11],
            '170926': [9, 10, 11],
            # '170928': [9, 10, 11],
        },

        'naive-hungry': {
            'train': [2, 3, 4],
            'running': [1],
            '170922': [5],
            '170925': [5],
            '170926': [5],
            # '170928': [5],
        },
    },

    'OA38': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            # '171016': [9, 10, 11],  # Early, stressed
            '171018': [9, 10, 11],
            '171023': [9, 10, 11],
            '171102': [9, 10, 11],
            '171107': [9, 10, 11],
            '171109': [9, 10, 11],
            '171110': [9, 10, 11],
            '171113': [9, 10, 11],
            '171114': [9, 10, 11],
            '171117': [9, 10, 11],
            '171121': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'photometry': ['nacc'],
            'running': [1],
            '171018': [5],
            '171023': [5],
            '171102': [5],
            '171107': [5],
            '171109': [5],
            '171110': [5],
            '171113': [5],
            '171114': [5],
            '171117': [5],
            '171121': [5],
        },

        'sated-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '171120': [9, 10, 11],
        },

        'hungry-no-phot': {
            'train': [2, 3, 4],
            'running': [1],
            '171120': [5],
        },
    },

    'OA39': {
        'sated-rotating-stimuli': {
            'train': [2, 3, 4],
            'running': [1],
            '171031': [9, 10, 11, 12],
            '171101': [9, 10, 11],
            '171108': [9, 10, 11],
        },

        'hungry-rotating-stimuli': {
            'train': [2, 3, 4],
            'running': [1],
            '171108': [5],
        },
    },

    'AS41': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '171130': [9, 10],
            '171201': [9, 10, 11],
            '171205': [9, 10, 11],
            '171207': [9, 10, 11],
            '171208': [9, 10, 11],
            '171211': [9, 10, 11],
            '171212': [9, 10, 11],
            '171214': [9, 10, 11],
            '171219': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'photometry': ['nacc'],
            'running': [1],
            '171130': [5],
            '171205': [5],
            '171207': [5],
            '171208': [5],
            '171211': [5],
            '171212': [5],
            '171214': [5],
            '171219': [5],
        },
    },

    'AS44': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180109': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'photometry': ['nacc'],
            'running': [1],
            '180109': [5],
        },
    },

    'AS46': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180116': [9, 10, 11],
            '180119': [10, 11, 12],
            '180124': [9, 10, 11],
        },

        'sated-water-loss': {
            'train': [2, 3],
            'running': [1],
            'photometry': ['nacc'],
            '180122': [9, 10, 11],
        },
    },

    'AS47-naive': {
        'naive-sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180129': [9, 10, 11],
            '180201': [9, 10, 11],
            '180202': [9, 10, 11],
            '180205': [9, 10, 11],
            # '180206': [9, 10, 11],  # Too much water loss
            '180208': [9, 10, 11],
            '180209': [9, 10, 11],
        },
    },

    'AS47': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180214': [9, 10, 11],
            '180216': [9, 10, 11],
            '180219': [9, 10, 11],
            '180221': [9, 10, 11],
            '180223': [9, 10, 11],
            # '180226': [9, 10, 11],
            '180228': [9, 10, 11],
            '180302': [9, 10, 11],
            '180305': [9, 10, 11],
            '180312': [9, 10, 11],
        },

        'hungry': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180214': [5],
            '180216': [5],
            '180219': [5],
            '180221': [5],
            '180223': [5],
            '180228': [5],
            '180302': [5],
            '180305': [5],
            '180312': [5],
        },
    },

    'AS51': {
        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180413': [9, 10, 11],
            '180419': [9, 10, 11],
            '180423': [9, 10, 11],
        },
    },

    'AS55': {
        'sated-testing': {
            'train': [1, 2],
            'running': [],
            'photometry': ['nacc'],
            '180702': [101],
        },

        'sated': {
            'train': [2, 3, 4],
            'running': [1],
            'photometry': ['nacc'],
            '180710': [6, 8],
        },
    },

    'OA178': {
        'sated': {
            'train': [2, 3],
            'running': [1],
            '180601': [9, 10, 11],
            '180612': [9, 10],
            '180614': [9, 10, 11],
            '180619': [9, 10, 11],
            # '180621': [9, 10, 11],
            '180625': [9, 10, 11],
            '180627': [9, 10, 11],
            '180629': [9, 10, 11],
            '180702': [9, 10, 11],
            '180706': [9, 10, 11],
            '180710': [9, 10, 11],
            '180712': [9, 10, 11],
            '180716': [9, 10, 11],
            '180718': [9, 10, 11],
        },
    },

    'OA191': {
        'sated': {
            'train': [2, 3],
            'running': [1],
            '180711': [9, 11],
            '180713': [9, 10, 11],
            '180717': [9, 10, 11],
            '180719': [9, 10, 11],
            '180723': [9, 10, 11],
            '180726': [9, 10, 11],
            '180728': [9, 10, 11],
            '180730': [9, 10, 11],
            # '180801': [9, 10, 11],
            '180803': [9, 10, 11],
            '180807': [9, 10, 11],
            '180809': [9, 10],
            '180813': [9, 10, 11],
            '180815': [9, 10, 11],
            '180817': [9, 10, 11],
            '180821': [9, 10, 11],
            '180823': [9, 10, 11],
            '180827': [9, 10, 11],
            '180829': [9, 10, 11],
            '180831': [9, 10],
            '180904': [9, 10, 11],
            # '180906': [9, 10, 11],
            # '180910': [9, 10, 11],
            # '180912': [9, 10, 11],

        },
    },

    'OA192': {
        'sated': {
            'train': [2, 3],
            'running': [1],
            '180802': [9, 10, 11],
            '180808': [9, 10, 11],
            '180814': [9, 10, 11],
            '180816': [9, 10, 11],
            '180820': [9, 10, 11],
            '180822': [9, 10, 11],
            # '180828': [9, 10, 11],
            '180830': [9, 10, 11],
            '180903': [9, 10, 11],
            '180905': [9, 10, 11],
            '180907': [9, 10, 11],
            '180911': [9, 10, 11],
            '180913': [9, 10, 11],
            '180917': [9, 10, 11],
            '180919': [9, 10, 11],
        },
        'sated-bad2': {
            'train': [3],
            'running': [1],
            '180806': [9, 10, 11]
        },
        'sated-bad3': {
            'train': [2],
            'running': [1],
            '180731': [9, 10, 11],
            '180810': [9, 10, 11]
        }
    },
}

sleep = {
    'AS21': {
        '161101': {
            5: [-1, -1],
            9: [-1, -1],
            10: [-1, -1],
            11: [-1, -1],
        },
        '161102': {
            5: [-1, -1],
            9: [-1, -1],
            10: [-1, -1],
            11: [-1, -1],
        },
        '161108': {
            5: [-1, -1],
            9: [-1, -1],
            10: [-1, -1],
            11: [-1, -1],
        },
        '161109': {
            5: [-1, -1],
            9: [0, 40],
            10: [0, 40],
            11: [0, 40],
        }
    }
}

reversals = {
    'CB173': '160516',
    'AS20': '160827',
    'AS21': '161115',
    'AS23': '161213',
    'OA32': '170406',
    'OA32-H': '181231',
    'OA34': '170602',
    'OA36': '170906',
    'OA37': '171231',
    'OA37-H': '171231',
    'OA38': '171231',
    'AS41': '181231',
    'AS44': '181231',
    'AS46': '181231',
    'AS47-naive': '181231',
    'AS47': '181231',
    'AS51': '181231',
    'OA178': '180702',
    'OA191': '180813',
    'OA192': '180903',
}


def reversal(mouse):
    """Return the date of the reversal of a given mouse.

    :param mouse: mouse name, string
    :return: date of reversal, string

    """
    if mouse not in reversals:
        return None
    return reversals[mouse]


def checkreversal(mouse, date, match='', optmatch=None):
    """Check whether a mouse and date are pre- or post-reversal.

    :param mouse: mouse name, str
    :param date: date, int or str
    :param match: match, str, 'pre', 'post', or anything else for both OR dict of kvargs to check if str optmatch exists
    :param optmatch: match, str, if match is a dict. Will be checked for in dict.
    :return: boolean

    """
    if optmatch is not None:
        if optmatch not in match:
            return True
        else:
            match = match[optmatch]

    if match.lower() == 'pre':
        return True if int(date) < int(reversals[mouse]) else False
    elif match.lower() == 'post':
        return True if int(date) >= int(reversals[mouse]) else False
    else:
        return True

def ids(mouse):
    """
    Get a list of all dates and days for each mouse sorted by cell ID.
    """

    # Make sure the mouse exists
    out = {}
    if mouse not in spontaneous: return []

    # Iterate through all of the entries
    for group in spontaneous[mouse]:
        runs = {}

        # Iterate once to get all extra (non-spontaneous) days
        for date in spontaneous[mouse][group]:
            if not date.isdigit():
                runs[date] = spontaneous[mouse][group][date]

        # Iterate a second time to get all
        for date in spontaneous[mouse][group]:
            if date.isdigit():
                runcopy = deepcopy(runs)
                runcopy['spontaneous'] = spontaneous[mouse][group][date]

                # Get the ID file
                ids = readids(mouse, date)

                # Add all IDs to the list
                for id in ids:
                    if id not in out:
                        out[id] = [(date, runcopy)]
                    else:
                        out[id].append((date, runcopy))

    # Save output as a sorted list of (id, date)
    import pdb;pdb.set_trace()
    sout = sorted([(key, sorted([out[key][i]][1])) for key in out])
    return sout

def sortedspontaneous():
    """
    Sort data from all spontaneous days and return all dates and days.
    """

    out = []
    for mouse in spontaneous:
        for group in spontaneous[mouse]:
            for date in spontaneous[mouse][group]:
                if date.isdigit():
                    for run in spontaneous[mouse][group][date]:
                        out.append((mouse, date, run, group))
    out = sorted(out, key=lambda x: (x[0], int(x[1]), int(x[2])))
    return out

def sortedall():
    """
    Sort data from all spontaneous days and return all dates and days.
    """

    addedmdrs = []
    out = []
    for mouse in spontaneous:
        for group in spontaneous[mouse]:
            groupdates = [date for date in spontaneous[mouse][group] if date.isdigit()]

            for date in groupdates:
                for run in spontaneous[mouse][group][date]:
                    out.append((mouse, date, run, group))

            for extra in ['train', 'running', 'sated-stim']:
                if extra in spontaneous[mouse][group]:
                    for run in spontaneous[mouse][group][extra]:
                        for date in groupdates:
                            if '%s-%s-%i'%(mouse, date, run) not in addedmdrs:
                                out.append((mouse, date, run, group))
                                addedmdrs.append('%s-%s-%i'%(mouse, date, run))
    out = sorted(out, key=lambda x: (x[0], int(x[1]), int(x[2])))

    return out


def dataframe(mice=None, dates=None, tags=None, runtypes=None, sort=False, groups=None):
    """Return a dataframe containing all metadata.

    Parameters
    ----------
    mice : list of str
        Mice to include.
    dates : list of int
        Dates to include.
    groups : list of str
        Deprecated. Groups to include.
    tags : list of str
        Tags to include.
    runtypes : list of str
        Runtypes to include.
    sort : bool
        If True, sort rows in dataframe by mouse, date, run.

    Returns
    -------
    pd.DataFrame

    """
    global _dataframe
    if _dataframe is None:
        out = []
        for mouse in spontaneous:
            for group in spontaneous[mouse]:
                groupdates = [
                    date for date in spontaneous[mouse][group] if date.isdigit()]
                for date in groupdates:
                    for run in spontaneous[mouse][group][date]:
                        out.append((mouse, date, run, group, 'spontaneous'))

                for extra in ['train', 'running', 'sated-stim']:
                    if extra in spontaneous[mouse][group]:
                        for run in spontaneous[mouse][group][extra]:
                            for date in groupdates:
                                out.append((mouse, date, run, group, extra))
        _dataframe = pd.DataFrame(
            out, columns=['mouse', 'date', 'run', 'group', 'runtype'])

    df = _dataframe.copy(deep=False)

    # Convert to numeric columns...should this happen?
    # If not, should probably do it in at least temporarily in the sort.
    df.date = pd.to_numeric(df.date)
    df.run = pd.to_numeric(df.run)

    # Start filtering
    if mice is not None:
        df = df[df.mouse.isin(mice)]
    if dates is not None:
        df = df[df.date.isin(dates)]
    if groups is not None:
        df = df[df.group.isin(groups)]
    if runtypes is not None:
        df = df[df.runtype.isin(runtypes)]

    if sort:
        df = df.sort_values(by=['mouse', 'date', 'run']).reset_index(drop=True)

    return df

def dates(mouse, grouplim=''):
    """
    Sort data from all spontaneous days of mouse mouse return all dates.
    """

    out = []
    if mouse in spontaneous:
        for group in spontaneous[mouse]:
            # Limit to the text in grouplim or all groups if empty
            if grouplim in group:
                for date in spontaneous[mouse][group]:
                    if date.isdigit():
                        out.append(date)

    # Sort and return
    out = sorted(list(set(out)))
    return out

def mice():
    """
    List all mice in metadata
    :return:
    """

    out = []
    for mouse in spontaneous:
        out.append(mouse)

    return sorted(out)

def mdr(mouse, date, run, matchgroup=''):
    """
    Return all information for a particular mouse, date, and run.
    """

    out = {}
    if mouse in spontaneous:
        fnd = False
        msgroups = [key for key in spontaneous[mouse]]
        if len(matchgroup) > 0:
            msgroups = [matchgroup] if matchgroup in msgroups else []

        for group in msgroups:
            if not fnd:
                if str(date) in spontaneous[mouse][group]:
                    if (run in spontaneous[mouse][group][str(date)] or
                        run in spontaneous[mouse][group]['train'] or
                        ('running' in spontaneous[mouse][group] and run in spontaneous[mouse][group]['running']) or
                        ('sated-stim' in spontaneous[mouse][group] and run in spontaneous[mouse][group]['sated-stim'])):

                        for metadata in spontaneous[mouse][group]:
                            if not metadata.isdigit():
                                out[metadata] = spontaneous[mouse][group][metadata]
                        if 'group' not in out:
                            out['group'] = [group]
                        else:
                            out['group'].append(group)

    out['mouse'] = mouse
    out['date'] = date
    for key in out:
        if isinstance(out[key], list):
            out[key] = sorted(list(set(out[key])))
    return out

def data(mouse, date):
    """
    Get all of the metadata for a mouse and date
    :param mouse: string
    :param date: string or 6-digit int, YYMMDD
    :return: dict of metadata
    """

    out = {}
    if mouse in spontaneous:
        for group in spontaneous[mouse]:
            if str(date) in spontaneous[mouse][group]:
                if 'spontaneous' not in out: out['spontaneous'] = []

                out['spontaneous'].extend(spontaneous[mouse][group][str(date)])
                for metadata in spontaneous[mouse][group]:
                    if not metadata.isdigit():
                        out[metadata] = spontaneous[mouse][group][metadata]
    out['hungry'], out['sated'] = hungrysated(mouse, date)
    return out

def hungrysated(mouse, date):
    """
    Return hungry and sated days for a mouse
    :param mouse:
    :param date:
    :param run:
    :return:
    """

    hungry = []
    sated = []
    if mouse in spontaneous:
        for group in spontaneous[mouse]:
            if str(date) in spontaneous[mouse][group]:
                if 'hungry' in group:
                    hungry.extend(spontaneous[mouse][group][str(date)])
                elif 'sated' in group:
                    sated.extend(spontaneous[mouse][group][str(date)])

    return sorted(list(set(hungry))), sorted(list(set(sated)))

def runs(mouse, date, grouplim=''):
    """
    Return all information for a particular mouse, date, and optionally, group.
    """

    out = {}
    # Limit to mice in spontaneous
    if mouse in spontaneous:
        for group in spontaneous[mouse]:
            # Limit to the text in grouplim or all groups if empty
            if grouplim in group:
                if str(date) in spontaneous[mouse][group]:
                    # Make an array we can extend
                    if 'spontaneous' not in out:
                        out['spontaneous'] = []

                    out['spontaneous'].extend(spontaneous[mouse][group][str(date)])
                    for metadata in spontaneous[mouse][group]:
                        if not metadata.isdigit():
                            out[metadata] = spontaneous[mouse][group][metadata]

    if 'spontaneous' in out:
        out['spontaneous'] = sorted(list(set(out['spontaneous'])))
    return out['spontaneous']

def md(mouse, date): return runs(mouse, date)


if __name__ == '__main__':
    # print md('AS23', '170131')
    # print mdr('AS23', '170131', 9)
    # print hungrysated('AS23', '170123')
    print(dates('CB173'))
    print(data('AS23', '161215'))
    print(data('OA32', '170412'))
