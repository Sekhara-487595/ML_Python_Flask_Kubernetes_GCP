from datetime import datetime
from pymongo import MongoClient
import os

class App_Logger:
    def __init__(self):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        self.dbName = 'AppLogging'
        #self.client = MongoClient('mongodb://localhost:27017') # connecting to mongodb which is running locally
        self.client = MongoClient('mongodb://mymongo:27017') # connecting to the mongodb which is running in container mymongo
        #self.client = MongoClient("mongodb+srv://test:test123@cluster0.nw74c.mongodb.net/mydatabase?retryWrites=true&w=majority") # connecting mongodb which running in mlab clould
        #self.client = MongoClient(os.getenv('MONGODB_URI','mongodb://localhost:27017'),connectTimeOutMS=30000)
        #self.client = MongoClient(os.environ['MONGODB_URI'],connectTimeOutMS=80000)
        db = self.client[self.dbName]
        self.collName = str(self.date) + "_" + 'logs'
        self.logs = db[self.collName]

    """def log(self, file_object, log_message):
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        file_object.write(str(date) + "/" + str(current_time) + "\t\t" + log_message + "\n")"""
    def logger_log(self,message):
        try:
            logs_data = {
                'date': str(self.date),
                'time': str(self.current_time),
                'log_message': message
            }
            self.logs.insert_one(logs_data)
        except ConnectionError:
            raise ConnectionError