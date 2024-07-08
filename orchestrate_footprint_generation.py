
from dotenv import load_dotenv
from analyze_solution_picture import analyze_picture
from generate_model_footprint import get_model_footprint

x=load_dotenv()
image_path="C:/Users/MANISHGUPTA/kyndryl/github/solution-sustainability/images/architecture.png"
#components = analyze_picture(image_path)
#print(components)
#exit()

components = ['Azure Sphere', 'IoT Hub', 'Azure Sphere Security Service', 'Azure Stream Analytics', 'Azure Cosmos DB', 'Azure SQL', 'Azure Synapse Analytics', 'Web and mobile apps', 'Power Platform and BI apps', 'API Management', 'Azure IoT Edge', 'Azure Defender for IoT', 'Azure DevOps', 'Azure Monitor', 'Azure Key Vault', 'Azure Active Directory', 'HoloLens']
print("------------------------------------------------------")
test_component = [components[len(components)-3]]
outputs = get_model_footprint(test_component)
print(outputs)


outputs = [
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
]

for output in outputs:
    print(output)