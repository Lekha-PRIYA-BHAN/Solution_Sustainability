from pymongo import MongoClient
import os
import csv
import json
import pandas as pd
from dotenv import load_dotenv
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define environment variables
def load_environment_variables():
    try:
        load_dotenv()
        db_host = os.getenv("DB_HOST")
        db_port = int(os.getenv("DB_PORT"))
        db_username = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        questions_csv = os.getenv("QUESTIONS_CSV")
        return db_host,db_port,db_username,db_password,db_name
    except Exception as e:
        logger.error("Error loading environment variables")
# Connect to MongoDB
def connect_to_db(db_host,db_port,db_username,db_password,db_name):
    try:
        db_client = MongoClient(f"mongodb://{db_username}:{db_password}@{db_host}:{db_port}/")
        db = db_client[db_name]
        return db_client,db
    except Exception as e:
        logger.error("Error creating db")
# Write the recommendations to the Database collection "Recommendations" with the project name as the document id
def write_recommendations_to_db(db,collection_name, project_name, recommendations):
    try:
        collection = db[collection_name]
        #collection.update_one({"Project": f"{project_name}"}, {"$set": {"Recommendations": recommendations}})
        collection.update_one({"Project": f"{project_name}"}, {"$set": {"Recommendations": recommendations}}, upsert=True)
    except Exception as e:
        logger.error("Error writing recommendations to db")

# Read recommendations from the DB collection "Recommendations" with the project name as the document id
def read_recommendations_from_db(db,collection_name, data_from):
    try:
        collection = db[collection_name]
        recommendations = collection.find_one({"Project": f"{data_from}"})
        return recommendations["Recommendations"]
    except Exception as e:
        error=logger.error(f"Error reading recommendations from db: {e}")
        return error

# Populate data to database
def populate_data_to_db(db,service, collection_name):
    try:
        collection = db[collection_name]
        input_file = (f"ai-report/{service}.json")  
        with open(input_file, 'r') as file:
            data = file.read()
            json_file = json.loads(data)
            db_data = json_file.get(f"{collection_name}", {})
            db_data["Component"] = f"{service}"
            #collection.insert_one(db_data)
            collection.update_one({"Component": f"{service}"}, {"$set": db_data}, upsert=True)
    except Exception as e:
        logger.error("Error populating data to db")

def read_data_from_db(db,collection_name,service):
    try:
        collection=db[collection_name]
        data = pd.DataFrame(list(collection.find({"Component": f"{service}"})))
        fields_to_exclude=["_id"]
        transposed_data = data.drop(fields_to_exclude,axis=1).set_index("Component").T
        
        #data_filterd = transposed_data.drop(fields_to_exclude, axis=1)
        # data_without_index = data_filterd.style.hide(axis="0")
        data_as_dataframe=pd.DataFrame(transposed_data)
        return data_as_dataframe
    except:
        logger.error("Error reading data from DB")

def check_service_in_db(db,collection_name,service):
    try:
        collection=db[collection_name]
        query = {"Component": f"{service}"}
        result = collection.find_one(query)
        return result is not None
    except:
        logger.error("Error reading data from DB")

def update_one_value(db, service, collection_name, field_to_update, value_to_update):
    try:
        collection = db[collection_name]
        collection.update_one({"Component": f"{service}"}, {"$set": {field_to_update: value_to_update}})
        return True
    except:
        logger.error("Error updating data in DB")
        return False
    
# # Main function
# if __name__ == "__main__":
#     # Load environment variables
#     db_host,db_port,db_username,db_password,db_name = load_environment_variables()
#     # Connect to database
#     db_client,db=connect_to_db(db_host,db_port,db_username,db_password,db_name)
#     # Load initial set of questions.
#     collection_name="Recommendations"
#     data = read_recommendations_from_db(db,collection_name,"Azure")
#     print(data)
#     service="Azure API Management"
#     data = read_data_from_db(collection_name,service)
#     print(data)
#     #populate_data_to_db(service, collection_name)


