import json


def get_dataset():
    with open('data/raw/dataset.json', 'r') as f:
        return json.load(f)
