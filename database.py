from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.pool import QueuePool
class Database:
    """
    Database access utilities
    """
    def __init__(self, host, port, schema, user, pasw):
        self.url = 'mysql+mysqlconnector://{user}:{pasw}@{host}:{port}/{schema}?charset=utf8'.format(
                user=user, pasw=pasw, host=host, port=port, schema=schema)
        self.engine = None
    def create_engine(self):
        if self.engine==None:
            try:
                self.engine = create_engine(self.url, pool_pre_ping=True, poolclass=QueuePool, pool_size=5, pool_recycle=-900, echo=False)
            except Exception as e:
                print("Couldn't create database engine - {}".format(str(e)))
        return self.engine
    def connection(self):
        return self.create_engine().connection()
    #FIXME Deprecated
    def get_client(self, file, connection):
        query = text("Select Client from ClientUserMapping WHERE User =:use")
        result = self.connection().execute(query, use=file.split('_')[1]).fetchone()
        return result if result!=None else 'Unknown'
    #FIXME Deprecated
    def get_database(self, file, connection):
        query = text("SELECT Client_Data_Base FROM ClientUserMapping WHERE User =:use")
        result = self.connection().execute(query, use=file.split('_')[1]).fetchone()
        return result
    #FIXME Deprecated
    def get_file_paths(self, path, connection):
        query = text("SELECT * FROM AutomationDirectoryInformation")
        result = self.connection().execute(query).fetchone()
        return result[path]