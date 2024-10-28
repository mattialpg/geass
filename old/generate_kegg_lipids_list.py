import json
import pandas as pd

with open('br08002.json', 'r') as json_file:
    lipid_data = json.load(json_file)

all_lipids = {}

def extract_lipids(obj):
    if "children" in obj:
        for child in obj['children']:
            extract_lipids(child)
    else:
        print(obj['name'])
        split_string = '  '
        if '  ' not in obj['name']:
            split_string = ' '
        id_and_name = obj['name'].split(split_string)
        all_lipids[id_and_name[0]] = id_and_name[1]

extract_lipids(lipid_data)

with open('kegg_lipids.json' , 'w') as res_file:
    json.dump(all_lipids, res_file)
