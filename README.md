# Solution / Sprint Plan ESG Analyzer

This app will help you estimate the carbon footprint of your architecture.
It will provide you with an estimate of the carbon footprint for each component and the total carbon footprint.

## Introduction

This application takes user inputs in the form of a Word document or PNG image. The application will analyz
This application is a Python-based project. The main file is `Home.py`.

## Usage

Once the application is setup successfully by following the steps described in the [Installation](#installation) section of this README,

* Open the Application UI [http://localhost:5000](http://localhost:5000)
* Toggle the “Run in demo mode” switch in the sidebar to simulate data instead of invoking Generative AI multiple times.
* Upload relevant data or input parameters.
* View the estimated carbon footprint.

## Requirements

You will need the following setup before running the application,

* Python 3.11 or higher
* Docker and Docker Compose
* An AWS account with access to Amazon Bedrock and Amazon S3

## Installation

* Clone this repository to your local machine.
* **IMPORTANT**: Modify the `.env` file to add the OPENAI API Key
* **IMPORTANT**: The application assumes that you are authenticated to AWS using the credentials stored in your `$HOME/.aws/credentials` file.
* Run the command docker-compose up -d to build and run the application.
* Open your browser and go to [http://localhost:5000](http://localhost:5000) to access the user interface.

## How to use the Code

Look at [orchestrate_footprint_generation.py](./orchestrate_footprint_generation.py).

It analyzes the architecture picture using `analyze_picture(..)` and the call generate a list of all the component names found.

For example a list of components:

```json
['Azure Sphere', 'IoT Hub', 'Azure Sphere Security Service', 'Azure Stream Analytics', 'Azure Cosmos DB', 'Azure SQL', 'Azure Synapse Analytics', 'Web and mobile apps', 'Power Platform and BI apps', 'API Management', 'Azure IoT Edge', 'Azure Defender for IoT', 'Azure DevOps', 'Azure Monitor', 'Azure Key Vault', 'Azure Active Directory', 'HoloLens']
```


The list will be passed as input to `get_model_footprint(...)` to generate the model and footprint for each of the components.
An example of it for "**Azure Key Vault**" component is shown below:

```json
{
  "Description of model": "This model estimates the carbon cloud footprint of using Azure Key Vault, a service that safeguards cryptographic keys and other secrets used by cloud applications and services. The model considers the energy consumption of the service, the data center's Power Usage Effectiveness (PUE), and the carbon intensity of the electricity supply to compute the carbon footprint.",
  "Model Parameters": {
    "EnergyConsumptionPerTransaction": "Energy consumed by Azure Key Vault per transaction in kWh.",
    "NumberOfTransactions": "Total number of transactions made by the user.",
    "PUE": "Power Usage Effectiveness of the Azure data center where Key Vault is hosted.",
    "CarbonIntensity": "Carbon intensity of the electricity supply to the data center, measured in kg CO2 per kWh."
  },
  "Assumptions": {
    "EnergyConsumptionPerTransaction": "0.000003 kWh per transaction, based on an estimated average for cryptographic operations and data retrieval.",
    "NumberOfTransactions": "1,000,000 transactions per month.",
    "PUE": "1.125, assuming a highly efficient Azure data center.",
    "CarbonIntensity": "0.4 kg CO2 per kWh, assuming a mix of renewable and non-renewable energy sources."
  },
  "Formulas": {
    "TotalEnergyConsumption": "EnergyConsumptionPerTransaction * NumberOfTransactions",
    "TotalEnergyConsumptionWithPUE": "TotalEnergyConsumption * PUE",
    "CarbonFootprint": "TotalEnergyConsumptionWithPUE * CarbonIntensity"
  },
  "Computation": {
    "TotalEnergyConsumption": "0.000003 kWh * 1,000,000 = 3 kWh",
    "TotalEnergyConsumptionWithPUE": "3 kWh * 1.125 = 3.375 kWh",
    "CarbonFootprint": "3.375 kWh * 0.4 kg CO2/kWh = 1.35 kg CO2"
  },
  "Conclusion": "Using Azure Key Vault for 1,000,000 transactions in a month results in a carbon footprint of approximately 1.35 kg CO2, under the given assumptions. This model highlights the importance of optimizing the number of transactions and selecting data centers with lower PUE and cleaner energy sources to minimize the carbon footprint of cloud services."
}
```

To save cost during testing I will suggest to choose to only pass one component to `get_model_footprint(...)`.

## License

The licensing model for this application is yet to be finalized.
