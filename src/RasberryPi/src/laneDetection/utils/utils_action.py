import json

def load_config_file(config_file):
    with open(config_file, "r") as jsonfile:
        data = json.load(jsonfile) # Reading the file
        print("Read successful")
        jsonfile.close()
    return data