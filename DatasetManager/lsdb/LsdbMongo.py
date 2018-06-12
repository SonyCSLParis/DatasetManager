import json
from glob import glob

from pymongo import MongoClient
from sshtunnel import SSHTunnelForwarder

from DatasetManager.lsdb.passwords import LOGIN_READONLY, PASSWORD_READONLY, SERVER_ADDRESS, \
    SSH_PKEY, SSH_USER_NAME


class LsdbMongo:
    def __init__(self):
        self.server = SSHTunnelForwarder(
            SERVER_ADDRESS,
            ssh_username=SSH_USER_NAME,
            remote_bind_address=('127.0.0.1', 27017),
            ssh_pkey=SSH_PKEY
        )
        self.server.start()

        self.client = MongoClient('localhost',
                                  self.server.local_bind_port,
                                  )

    def get_db(self):
        # Cannot connect directly to lsdb and then performing auth
        db = self.client.get_database('admin')
        db.authenticate(LOGIN_READONLY, PASSWORD_READONLY,
                        mechanism='SCRAM-SHA-1')
        lsdb = self.client.get_database('lsdb')
        return lsdb

    def get_songbook_leadsheets_cursor(self, db):
        """Return a cursor all songbook leadsheets, excluding user input ones
        """
        return db.leadsheets.find(
            {'source': {"$ne": "51b6fe4067ca227d25665b0e"}})

    def close(self):
        self.client.close()
        self.server.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    lsdb_client = LsdbMongo()
    db = lsdb_client.get_db()
    # cursor = lsdb_client.get_songbook_leadsheets_cursor(db)
    # print(next(cursor))
    print(db.find())
    lsdb_client.close()
