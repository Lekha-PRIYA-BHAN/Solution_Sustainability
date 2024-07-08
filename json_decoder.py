import json
import re 



# Read a text file containing json and store only the JSON part in another file.
service="Azure IoT Edge"
def json_decoder(service):
    input_file = f'ai-report/{service}.txt'
    output_file = f'ai-report/{service}.json'
    with open(input_file, 'r') as file:
        data = file.read()
        json_data = json.loads(data[data.index('{'):])
        # Save the json_data to an output file
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=4)

def json_decoder1(service, emission_report):
    emmission_text_file = f'ai-report/{service}.txt'
    output_json_file = f'ai-report/{service}.json'
    # Save the emission_report to a text file
    with open(emmission_text_file, 'w') as file:
        file.write(emission_report)
    with open(emmission_text_file, 'r') as file:
        data = file.read()
        json_data = json.loads(data[data.index('{'):])
    # Save the json_data to an output file
    with open(output_json_file, 'w') as file:
        json.dump(json_data, file, indent=4)

def json_decoder2(service):
    input_file = f'ai-report/{service}.txt'
    output_file = f'ai-report/{service}.json'
    json_pattern = r"({.+?})"
    with open(input_file, 'r') as file:
        data = file.read()
        json_matches = re.findall(json_pattern, data, re.DOTALL)
        s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
        print(json_matches)
        if json_matches:
            extracted_json = json_matches[1]
            print(extracted_json)
            # Parse the extracted JSON
            try:
                parsed_json = json.loads(extracted_json)
                print(parsed_json)
                # Save the parsed JSON to a file
                with open(output_file, "w") as json_file:
                    json.dump(parsed_json, json_file, indent=4)
                print(f"Extracted JSON content saved to {output_file}.")
            except json.JSONDecodeError:
                print("Error: Unable to parse the extracted JSON content.")
        else:
            print("No JSON-like content found in the emmission report.")

# A function to decode the JSON content from a text file where the start is JSONSTART and the end is JSONEND
def json_decoder3(service):
    input_file = f'ai-report/{service}.txt'
    output_file = f'ai-report/{service}.json'
    json_start = "JSONSTART"
    json_end = "JSONEND"
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

# json_data = json_decoder3(service)
