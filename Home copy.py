# Import Modules for the program
import streamlit as st  # StreamLit UI
import base64           # Used for converting picture
import boto3            # Used by AWS LLM
import json             # Used for Parsing JSON files
import tempfile         # Used for storing files temporarily during upload
import pandas as pd     # Used for Data Analysis
import re               # Used for Regular Expression
import time             # Used for Time
import plotly.express as px # Used for Data Visualization
import os
from dotenv import load_dotenv  # Used for loading .env file
from langchain_core.prompts import ChatPromptTemplate           # Used for Generating the Chat Prompt
from langchain_openai.chat_models import ChatOpenAI             # OpenAI LLM
from langchain.llms.bedrock import Bedrock                      # AWS Bedrock LLM
from langchain_core.messages import HumanMessage, SystemMessage # Parsing PROMPT
from langchain_community.document_loaders import Docx2txtLoader # Convert Word Document to Text
import database as database              # Used for Database Connection
import numpy as np

# Set the StreamLit UI to full wide display
st. set_page_config(layout="wide")

# Remove the default Streamlit header coloured line.
st.markdown( ''' <style> header {visibility: hidden;} </style>''',unsafe_allow_html=True)

# Application Title
st.title("Solution / Sprint Plan ESG Analyzer")
st.markdown("""This app will help you estimate the carbon footprint of your architecture.
            It will provide you with an estimate of the carbon footprint for each component and the total carbon footprint.
            """)

# Set the Streamlit left pane Caption

st.sidebar.image("datasources/kyndryl_logo.png", width=150)
st.sidebar.title("GreenSpark Hackathon 🌱✨")
st.sidebar.markdown ('Welcome to the **Sustainable World** 🌿')
demo_mode = st.sidebar.toggle('Run in demo mode', True)
st.sidebar.markdown("*To conserve resources and reduce expenses, demo mode will employ simulated data instead of invoking Generative AI multiple times.*")
app_type = st.sidebar.radio("Select the application type.", ["Solution Analyzer", "Sprint Plan Analyzer"])


# Get the model type from the user
#model_type = st.sidebar.radio("Select the LLM model to use.", [ "Amazon Bedrock (Only Text)", "OpenAI (Text & Image)"])

# Import variables from .env file.
x=load_dotenv()

# Local Variables are defined here
service="" # Setting a default value for the service
input_document = "" # Setting a default value for the prompt input document

# Load database
db_host,db_port,db_username,db_password,db_name = database.load_environment_variables()
db_client,db=database.connect_to_db(db_host,db_port,db_username,db_password,db_name)

# Calling the Bedrock LLM
def call_bedrock_model(prompt):
    try:
        model_id = "anthropic.claude-v2:1"  # Amazon Bedrock Anthropic Claude LLM
        model_kwargs = {                    # Amazon Bedrock Anthropic Claude LLM parameters
            "max_tokens_to_sample": 4000,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"]
           }
        llm = Bedrock(model_id=model_id, model_kwargs = model_kwargs)  # AWS Bedrock LLM
        llm_response = llm.invoke(prompt)
        return llm_response
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)

# Converted the Word Document uploaded from file_uploader() function to text
def convert_word_document_to_text(file_path):
    try:
        loader = Docx2txtLoader(file_path=file_path)
        content = loader.load()
        return content
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)

# Encode the Image uploaded from file_uploader() function
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)

# Get Architecture components using OpenAI LLM
def get_architecture_components_from_image(image_path):
    
    with open("./system_message.txt", "r") as f:
        system_message = f.read()
    
    with open("./human_message_2.txt", "r") as f:
        human_message = f.read()

    # Getting the base64 string
    base64_image = encode_image(image_path)

    messages = [
        SystemMessage(
            content=(
                {system_message}
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", 
                "text": f"""{human_message}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto",
                    },
                },
            ]
        )]

    llm = ChatOpenAI(temperature=0.3, model="gpt-4-vision-preview", max_tokens=3000, )
    chain =  ChatPromptTemplate.from_messages(messages)  | llm
    response = chain.invoke({})
    
    def getSubstringBetweenTwoChars(ch1,ch2,s):
        return s[s.find(ch1)+1:s.find(ch2)]
    x= getSubstringBetweenTwoChars('[', ']', response.content).replace("\n", "")
    components=[]
    for component in x.split(","):
        components.append(component.strip().replace("\"", ""))
    return components

# Get Architecture components using Amazon Bedrock
def get_architecture_components_from_word(file_path):
    try:
        input_document = convert_word_document_to_text(file_path)
        # Prompt for Listing Architecture Components using Amazon Bedrock
        prompt_for_listing_architecture_components = f"""
            Human: Findout and list the Azure cloud solution components given in the following document.
            {input_document}
            The output must be in a simple JSON array format, only in valid JSON format to allow for digital processing
            Don't provide any additional textual explanation."
            """
        llm_response = call_bedrock_model(prompt_for_listing_architecture_components)
        start_index = llm_response.find("[")
        end_index = llm_response.find("]") + 1
        architecture_components = llm_response[start_index:end_index]
        architecture_components = json.loads(architecture_components)
        return architecture_components
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)

# Get Architecture components using Dummy Values
def get_architecture_components_from_doc_dummy_values():
    time.sleep(1)
    return ["Azure Sphere", "Azure IoT Edge", "Azure IoT Hub", "Azure Stream Analytics (ASA)", "Azure Cosmos DB", "Azure SQL Database", "Azure Synapse Analytics", "Azure Synapse Link for Azure Cosmos DB", "Microsoft Power BI", "Azure App Services", "Azure API Management", "Microsoft HoloLens"]
    #return ["Azure Sphere", "Azure IoT Edge", "Azure IoT Hub", "Azure Stream Analytics (ASA)", "Azure Cosmos DB"]
def get_architecture_components_from_image_dummy_values():
    time.sleep(1)
    return ["Azure API Management", "Azure Active Directory", "Azure App Services", "Azure Cosmos DB","Azure Defender for IoT", "Azure DevOps", "Azure IoT Edge", "Azure IoT Hub", "Azure Key Vault", "Azure Monitor", "Azure SQL Database", "Azure Sphere Security Service", "Azure Sphere", "Azure Stream Analytics (ASA)", "Azure Synapse Analytics", "Azure Synapse Link for Azure Cosmos DB", "Azure Web Apps", "Microsoft HoloLens", "Microsoft Power BI"]

# Upload file to process
def file_uploader():
    try:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name
                file_type = "Word Document"
        else:
            st.sidebar.image(uploaded_file, caption="Uploaded Diagram", use_column_width=True)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name
                file_type = "Image"
        return uploaded_file, file_path, file_type
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)

# Extract the Emission number from paragraph
def extract_emission_number(paragraph):
    numeric_values = re.findall(r'"Total Emission":\s*"([0-9,.]+)"', paragraph)
    cleaned_values = []
    # Sometime the model returns emission value with some additional text. 
    # Extracting the numarical value from the text and making it a Float.
    for value in numeric_values:
        cleaned_value = value.replace(',', '')  # Remove commas
        try:
            cleaned_values.append(float(cleaned_value))  # Convert to float
        except ValueError:
            pass  # Ignore non-numeric values
    return cleaned_values

# Generate JSON file from the text file
def json_decoder(service, emission_report):
    input_file = f'ai-report/{service}.txt'
    output_file = f'ai-report/{service}.json'
    json_start = "JSONSTART"
    json_end = "JSONEND"
    # Save the emission_report to a text file
    if not os.path.exists(input_file):
        with open(input_file, 'w') as file:
            file.write(emission_report)
    with open(input_file, 'r') as file:
        data = file.read()
        json_start_index = data.find(json_start)
        json_end_index = data.find(json_end)
        if json_start_index != -1 and json_end_index != -1:
            json_data = data[json_start_index + len(json_start):json_end_index]
            # Save the json_data to an output file
            with open(output_file, 'w') as file:
                file.write(json_data)
        else:
            print("No JSON content found in the emmission report.")
# Generate Emission report for the Architecture components using Amazon Bedrock
def generate_emission_report(service):
    #Prompt for Generating Carbon Footprint using Amazon Bedrock
    prompt_for_generating_carbon_footprint = f"""
        Human:
        For the cloud service, named: {service}, provide a detailed model of how to compute an estimate of the carbon cloud footprint (in kg CO2 per month) leveraging the knowledge about the service. Follow the steps provided below to generate an output in json:
        Step 0: Think about a detailed model for computation of the carbon footprint of the {service}. You are not bound by the the examples but here are some example questions to consider: does the service have transactions; does it have to receive data, process data, and return data in these transactions; will there be significant energy consumption in data storage; will the service be running in high availability mode and thus having a higher level of energy consumption; think about the co2_emission_factor relevant in the model; etc etc.
        Step 1: Identify all the input parameters that will be used for the detailed model thought in Step 0. Ensure that the names are concisely defined. Note there will be only one output parameter which is "carbon_footprint" which will be the carbon footprint of the service. A parameter is to be considered only if in the subsequent steps it is used in constructing the formula.
        Step 2: Define a formula that would use the parameters in the Step 1. The names of the parameters in the formula must be the ones defined in Step 1.
        Step 3: Make your own required assumptions for the parameters of Step 1 and plug them in the formula of Step 2.
        Step 4: Generate a json as output, with the json containing and only containing the following keys in the order specified below:
                "Description": will describe the detailed model for the {service} thought in Step 0,
                "Model Parameters": will be an object containing each of the parameters (in Step 1) as keys and their corresponding values as their descriptions, including the output parameter 'carbon_footprint',
                "Units": will be the units for each of the parameters defined under 'Model Parameters' - set the value as 'ratio' if the parameter is a ratio; if a transaction-like parameter is included in Model Parameters then ensure that the unit is a rate, for example, 'transactions per month',
                "Formula": will be an object containing the formula of the Step 2, and in fact will have the starting word as 'carbon_footprint',
                "Assumptions": will be an object containing the assumption made in Step 3, and,
                "Carbon Footprint": will be an object containing the computation of the formula in Step 2 based on the assumptions in Step 3. Show the final computed value as 'value', and as well as show how it was computed based on the formula as 'computation'.

        Notes: 
        - Provide a valid json in Step 4 above.
        - Ensure that if a parameter is included under 'Model Parameters' then it is also used in the formula defined under 'Formula'
        - Ensure that units on both sides of the formula under 'Formula' is consistent
        - Provide only values as part of the Assumptions and do not add the units along with the values
        - If the value is a number, then provide it as a number and not as a string and do not include any commas in the number
        - The JSON content start with JSONSTART and ends with JSONEND
        Assistant:
        """
    # Call Bedrock for generating the report
    emission_report = call_bedrock_model(prompt_for_generating_carbon_footprint)
    return emission_report

# Generate Emission report for the Architecture components using Dummy Values
def generate_emission_report_dummy(service):
    time.sleep(1)
    try:
        file_path = f'ai-report/{service}.txt'
        with open(file_path, 'r') as file:
            emission_report = file.read()
    except Exception as e:
        st.error("Error: " + str(e))
        return "Error: " + str(e)
    return emission_report

# Calculate total Emission for the Architecture components
def calculate_total_emission_for_service(emission_report):
    try:
        total_emission_match = re.search(r'"Total Emission": (\d+)', emission_report)
        if total_emission_match:
            total_emission_value_for_service = int(total_emission_match.group(1))
        else:
            total_emission_value_for_service = 0 
    except ValueError:
        total_emission_value_for_service = 0
    return total_emission_value_for_service

# Main function
if __name__ == "__main__":
    if app_type == "Sprint Plan Analyzer":
        st.warning("This feature is not yet available. Please check back later.")
    elif app_type == "Solution Analyzer":
        uploaded_file = st.sidebar.file_uploader(f"Upload your architecture here.  (Either Word Document or an Architecture Diagram)", type=["png", "docx"])
        if uploaded_file is not None:
            uploaded_file, file_path, file_type = file_uploader()
            if st.sidebar.button("Analyze", type="primary"):
                if file_type == "Image":
                    # Get the architecture components using OpenAI LLM from the Image.
                    st.sidebar.markdown("Using OpenAI Model for Image Analysis")
                    with st.spinner('Getting the Architecture Components...'):
                        if demo_mode:
                            # Get the architecture components using Dummy Values
                            architecture_components = get_architecture_components_from_image_dummy_values()
                        else:
                            # Get the architecture components using OpenAI LLM
                            architecture_components = get_architecture_components_from_image_dummy_values()
                            #architecture_components = get_architecture_components_from_image(file_path)
                elif file_type == "Word Document":
                    # Get the architecture components using Amazon Bedrock LLM from the Word document.
                    st.sidebar.markdown("Using Amazon Model for Text Analysis")
                    with st.spinner('Getting the Architecture Components...'):
                        if demo_mode:
                            # Get Dummy values instead of making multiple calls to LLM to avoid Cost
                            architecture_components = get_architecture_components_from_doc_dummy_values()
                        else:
                            # Get the architecture components using Amazon Bedrock LLM
                            architecture_components = get_architecture_components_from_word(file_path)

                # Using the Architecture compoents generated, findout the Carbon Footprint using Amazon Bedrock
                total_architecture_emission = 0
                x_values = []
                y_values = []
                st.sidebar.markdown("## Detailed Emission report per component")
                for service in (architecture_components):
                    with st.spinner(f'Generating Carbon Footprint for {service}...'):
                        if demo_mode:
                            # Generate an emission report per service using Dummy Values
                            emission_report = generate_emission_report_dummy(service)
                        else:
                            # Generate an emission report per service using Amazon Bedrock
                            if database.check_service_in_db(db,"Assumptions",service):
                                alert=st.info(f"Data for {service} already exists in the database. Avoding Gen-AI call", icon="ℹ️")
                                time.sleep(2)
                                alert.empty()
                                emission_report = generate_emission_report_dummy(service)
                            else:
                                emission_report = generate_emission_report(service)
                                for collection_name in ["Model Parameters","Units","Formula","Assumptions","Carbon Footprint"]:
                                    # Populate the data to the database
                                    database.populate_data_to_db(db, service, collection_name)
                        # Save the report to a file
                        json_decoder(service, emission_report)
                        with st.sidebar.expander(f"Carbon Footprint Analysis for {service}"):
                            #st.markdown(f"{emission_report}")
                            for collection_name in ["Model Parameters","Units","Formula","Assumptions","Carbon Footprint"]:
                                component_data=database.read_data_from_db(db,collection_name,service)
                                st.markdown(f"### {collection_name}")
                                st.dataframe(component_data)
                                # if collection_name=="Carbon Footprint":
                                #     total_emission_value_for_service = component_data.loc["value"][0]
                                #     st.markdown(f"### Total Forecasted emission for {service} for Month:\n# {total_emission_value_for_service} kg CO2e")
                            formula=database.read_data_from_db(db,"Formula",service)
                            formula_for_service = formula.loc["carbon_footprint"].iloc[0]
                            assumptions = database.read_data_from_db(db,"Assumptions",service)
                            assumptions_for_service = pd.DataFrame(assumptions.T)
                            # Extract parameter names from the equation
                            parameter_names = re.findall(r'\b\w+\b', formula_for_service)
                            # Calculate emission from equation
                            def evaluate_dynamic_equation(row):
                                try:
                                    param_values = {param: row[param] for param in parameter_names if param in row}
                                    result = eval(formula_for_service, param_values)
                                    return result
                                except Exception as e:
                                    return np.nan
                            calculated_emission_for_service = assumptions_for_service.apply(evaluate_dynamic_equation, axis=1)
                            st.markdown(f"### Calculated Emission for {service} for Month:\n# {calculated_emission_for_service[0]} kg CO2e")
                        # Extract the Total emission for service from the report
                        total_emission_value_for_service = calculated_emission_for_service[0]
                        x_values.append(total_emission_value_for_service)
                        y_values.append(f"{service}")
                emission_dataframe = pd.DataFrame({'Emission': x_values, 'Service': y_values})
                # Calculate total emission for the Architecture components
                total_emission = emission_dataframe["Emission"].sum()
                # Display the Emission report
                emission_report_table= emission_dataframe[['Service', "Emission"]]
                markdown_list = "\n".join(f"- {Service}" for Service in emission_dataframe['Service'])
                emission_bar_chart = px.bar(emission_dataframe, x='Emission', y='Service', title='Emission by Components')
                emission_pie_chart = px.pie(emission_dataframe, values='Emission', names='Service', title='Emission percentage by Components', hole=.3)
                st.markdown(f"### Total Forecasted emission for your solution for 1 Month:\n# {total_emission} kg CO2e")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(emission_bar_chart)
                with col2:
                    st.plotly_chart(emission_pie_chart)
                st.markdown(f"### Solution Components Identified:\n")
                st.table(emission_report_table)

                # Provide a edit field to update the assumptions
                if st.button ("Provide Recommendations", type="primary"):
                    st.write("Provide the recommendations here")
                    recommendations=st.text_area("Recommendations")