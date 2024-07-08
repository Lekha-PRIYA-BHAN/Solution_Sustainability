import database as database  
import pandas as pd
import numpy as np
import re
# Load database
db_host,db_port,db_username,db_password,db_name = database.load_environment_variables()
db_client,db=database.connect_to_db(db_host,db_port,db_username,db_password,db_name)
# Receive formula and assumptions data from database
component = database.read_data_from_db(db,"test","Component 1")
assumption = component.loc["Assumptions"].values[0]
assumptions_df = pd.DataFrame(assumption, index=[0])
formula = component.loc["Formula"].values[0]
print(f"----------------------\nComponent: \n{component}\n----------------------")
print(f"----------------------\nAssumption: \n{assumptions_df.T}\n----------------------")
print(f"----------------------\nFormula: \n{formula}\n----------------------")