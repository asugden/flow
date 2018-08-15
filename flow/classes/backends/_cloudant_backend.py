"""Initial attempts at replacing couchdb library with cloudant library.

Incomplete.

"""
from cloudant.client import CouchDB, CloudantDatabaseException
from uuid import uuid4


class Connection(object):
    def __init__(self, host='localhost', port=5984):
        raise NotImplementedError
        username = 'USER'
        password = 'PASSWORD'

        self._connection = {}
        self._connection['host'] = host
        self._connection['port'] = port

        self._client = CouchDB(username, password, url='http://{}:{}'.format(
            self._connection['host'], self._connection['port']),
            connect=True, auto_renew=True)

    @property
    def client(self):
        return self._client

    def database(self, name):
        return Database(self.client, name)

class Database(object):
    def __init__(self, client, name):
        self._client = client
        self._db = self._client[name]

    def put(self, _id=None, **data):
        """Store a value in the database."""
        # new_data, numpy_array = self._numpy_put_prep(data)
        _id, _rev = self._put_assume_new(_id, **data)
        # if numpy_array is not None:
        #     doc = self._db.get(_id)
        #     temp_file = TemporaryFile()
        #     np.save(temp_file, numpy_array)
        #     self._db.put_attachment(doc, temp_file, filename='value')
        return _id, _rev

    @staticmethod
    def _numpy_put_prep(data):
        val = data.get('value', None)
        if isinstance(val, np.ndarray):
            data['value'] = '__attachment__'
        return data, val

    def _put_check_first(self, _id=None, **data):
        """Store a value in the database.

        Checks to see if _id already exists.

        """
        if _id is None:
            _id = str(uuid4())
        doc = dict(_id=_id, **data)
        if _id in self._db:
            current_doc = self.get(_id)
            doc['_rev'] = current_doc.rev
            return self._db.save(doc)
        else:
            return self._db.save(doc)

    def _put_assume_new(self, _id=None, **data):
        """Store a value in the database.

        Trys to immediately add doc and falls back to replace if needed.

        """
        if _id is None:
            _id = str(uuid4())
        doc = dict(_id=_id, **data)
        try:
            self._client.create_document(doc)
        except CloudantDatabaseException:
            
        # try:
        #     _id, _rev = self._db.save(doc)
        # except couchdb.http.ResourceConflict:
        #     # TODO: _rev is in header, don't need to get entire doc
        #     current_doc = self.get(_id)
        #     doc['_rev'] = current_doc.rev
        #     _id, _rev = self._db.save(doc)
        # return _id, _rev

    def _put_assume_replace(self, _id=None, **data):
        """Store a value in the database.

        Trys to get existing doc first, falls back to directly adding if int
        doesn't already exist.

        """
        if _id is None:
            _id = str(uuid4())
        doc = dict(_id=_id, **data)
        try:
            current_doc = self.get(_id)
        except couchdb.http.ResourceNotFound:
            _id, _rev = self._db.save(doc)
        else:
            # TODO: _rev is in header, don't need to get entire doc
            doc['_rev'] = current_doc.rev
            _id, _rev = self._db.save(doc)
        return _id, _rev

    def _put_delete(self, _id=None, **data):
        """Store a value in the database.

        Trys to delete a pre-existing entry first and then add.

        """
        if _id is None:
            _id = str(uuid4())
        doc = dict(_id=_id, **data)
        try:
            del self._db[_id]
        except couchdb.http.ResourceNotFound:
            pass
        return self._db.save(doc)

    def get(self, _id):
        """Return the value from the data store for a given analysis."""
        doc = self._db.get(_id, default=None)
        if doc['value'] == '__attachment__':
            doc['value'] = np.load(self._db.get_attachment(doc, 'value'))
        return doc


def test():
    pass

if __name__ == '__main__':
    test()