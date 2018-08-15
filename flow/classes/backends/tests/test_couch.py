from numpy.testing import run_module_suite, assert_equal
import numpy as np

from replay.lib.dep.backends.couch_backend import CouchBackend

host = 'localhost'
database = 'testing'

class TestCouchDB(object):
    def setup(self):
        self.db = CouchBackend(host=host, database=database)
        self.keys = {'updated': '180413',
                     'date': 180101,
                     'mouse': 'TM001'}

    def teardown(self):
        del self.db

    def test_float(self):
        analysis_name = 'test_float'
        val = 1.0
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_np_float(self):
        analysis_name = 'test_np_float'
        val = np.float64(1.0)
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_int(self):
        analysis_name = 'test_int'
        val = 1
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_np_int(self):
        analysis_name = 'test_np_int'
        val = np.int64(1)
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_str(self):
        analysis_name = 'test_str'
        val = 'test string'
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_list(self):
        analysis_name = 'test_list'
        val = [1, 2., 'foo', np.int64(1), np.float64(2.)]
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_array(self):
        analysis_name = 'test_array'
        val = np.array([1, 2., 'foo', np.int64(1), np.float64(2.)])
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan(self):
        analysis_name = 'test_nan'
        val = np.nan
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan_list(self):
        analysis_name = 'test_nan_list'
        val = [np.nan, 1, 2.]
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

    def test_nan_array(self):
        analysis_name = 'test_nan_array'
        val = np.array([np.nan, 1, 2.])
        self.db.store(analysis_name=analysis_name, data=val, keys=self.keys)
        val2 = self.db.recall(analysis_name=analysis_name, keys=self.keys)
        assert_equal(val2, val)

if __name__ == "__main__":
    run_module_suite()
