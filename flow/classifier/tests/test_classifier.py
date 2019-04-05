from numpy.testing import \
    run_module_suite, assert_, assert_allclose, assert_equal

import flow

mdr = {'mouse': 'AS41', 'date': 171130, 'run': 9}
run = None

orig, pars, out = None, None, None


def setup():
    """Setup."""
    global orig, pars, out, run

    orig = flow.misc.loadmat('data/AS41_171130_009.mat')

    pars = flow.config.default()
    run = flow.Run(mouse=mdr['mouse'], date=mdr['date'], run=mdr['run'])
    date = run.parent
    training_runs = date.runs(run_types=['training'])
    running_runs = date.runs(run_types=['running'])

    model, params, nan_cells = flow.classifier.train.train_classifier(
        run=run, training_runs=training_runs, running_runs=running_runs,
        training_date=date, verbose=False)
    out = flow.classifier.train.classify_reactivations(
        run=run, model=model, params=params, nan_cells=nan_cells)
    out['parameters'] = flow.misc.matlabifypars(out['parameters'])


def teardown(self):
    """Teardown."""
    pass


class TestClassifier(object):
    """Test classifier output to make sure it matches old data.

    Added in only looking after lastonset() to a few of the tests since
    we've tweaked how stimuli are masked by default.

    """

    def test_parameters(self):
        # This test allows for new parameters to be added, but all of the
        # original keys must be present. If this test fails, all the rest
        # probably should as well.
        def compare_dict(orig_d, test_d):
            for key in orig_d:
                assert_(
                    key in test_d,
                    msg="'{}' not in test parameters.".format(key))
                if isinstance(orig_d[key], dict):
                    assert_(
                        isinstance(test_d[key], dict),
                        msg="'{}' should be a dict.".format(key))
                    compare_dict(orig_d[key], test_d[key])
                # Let the classifier updated date change
                if key == 'classifier_updated_date':
                    continue
                # hack to deal with Matlab compressing away singleton
                # dimensions.
                if isinstance(test_d[key], list) and len(test_d[key]) == 1:
                    assert_equal(test_d[key], [orig_d[key]], err_msg=key)
                else:
                    assert_equal(test_d[key], orig_d[key], err_msg=key)

        compare_dict(orig['parameters'], out['parameters'])

    def test_priors(self):
        last_onset = run.trace2p().lastonset()
        for key in orig['priors']:
            assert_equal(out['priors'][key][last_onset:],
                         orig['priors'][key][last_onset:], err_msg=key)

    def test_result_keys(self):
        assert_equal(
            sorted(out['results'].keys()), sorted(orig['results'].keys()))

    def test_results_pnm(self):
        last_onset = run.trace2p().lastonset()
        for key in orig['results']:
            if key not in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['results'][key][last_onset:],
                orig['results'][key][last_onset:], rtol=0, atol=1e-4,
                err_msg=key)

    def test_results_other(self):
        last_onset = run.trace2p().lastonset()
        for key in orig['results']:
            if key in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['results'][key][last_onset:],
                orig['results'][key][last_onset:], rtol=0, atol=1e-4,
                err_msg=key)

    def test_marginal_keys(self):
        assert_equal(
            sorted(out['marginal'].keys()), sorted(orig['marginal'].keys()))

    def test_marginal_pnm(self):
        for key in orig['marginal']:
            if key not in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['marginal'][key], orig['marginal'][key],
                rtol=0, atol=1e-10, err_msg=key)

    def test_marginal_other(self):
        for key in orig['marginal']:
            if key in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['marginal'][key], orig['marginal'][key],
                rtol=0, atol=1e-10, err_msg=key)

    def test_likelihood_keys(self):
        assert_equal(
            sorted(out['likelihood'].keys()),
            sorted(orig['likelihood'].keys()))

    def test_likelihood_pnm(self):
        for key in orig['likelihood']:
            if key not in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['likelihood'][key], orig['likelihood'][key],
                rtol=0, atol=1e-10, err_msg=key)

    def test_likelihood_other(self):
        for key in orig['likelihood']:
            if key in ['plus', 'neutral', 'minus']:
                continue
            assert_allclose(
                out['likelihood'][key], orig['likelihood'][key],
                rtol=0, atol=1e-10, err_msg=key)

    def test_cellmask(self):
        assert_equal(out['cell_mask'], orig['cell_mask'])

if __name__ == '__main__':
    run_module_suite()
