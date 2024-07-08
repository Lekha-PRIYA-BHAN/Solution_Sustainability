import database as database  
import pandas as pd
import numpy as np
import re
# Load database
db_host,db_port,db_username,db_password,db_name = database.load_environment_variables()
db_client,db=database.connect_to_db(db_host,db_port,db_username,db_password,db_name)
# Receive formula and assumptions data from database
service = "Azure Cosmos DB"
formula=database.read_data_from_db(db,"Formula",service)
data=database.read_data_from_db(db,"Assumptions",service)
equation = formula.loc["carbon_footprint"].iloc[0]
df = pd.DataFrame(data.T)
# Extract parameter names from the equation
parameter_names = re.findall(r'\b\w+\b', equation)
def evaluate_dynamic_equation(row):
    try:
        print(f"----------------------\nRow: \n{row}\n----------------------")
        param_values = {param: row[param] for param in parameter_names if param in row}
        CalculatedValue = eval(equation, param_values)
        return CalculatedValue
    except Exception as e:
        return np.nan
df['Calculated_Value'] = df.apply(evaluate_dynamic_equation, axis=1)
CalulatedValue = df['Calculated_Value'].values[0]
print(f"----------------------\nEquation: \n{equation}\n----------------------")
print(f"----------------------\nAssumption Data & Result: \n{df.T}\n----------------------")
print(f"----------------------\nResult: {CalulatedValue} \n----------------------")