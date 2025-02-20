import os
import shutil
import json
from parse_sparql import *

parser = Parser()

def isdirect(question):
    if question["question-type"].startswith("Simple Question"):
        return question["question-type"][17:23] == "Direct"
    if "description" in question:
        if "ndirect" in question["description"]:
            return False
        if "ncomplete" in question["description"]:
            return False
    return True

def generate_train_data(file_path):
    data = json.load(open(file_path, 'r'), strict=False)
    processed_data = []
    for i in range(0, len(data), 2):
        question = data[i]
        answer = data[i + 1]
        qa = {}
        if not isdirect(question) and i > 0:
            context = context + data[i - 2]["utterance"] + " [SEP] " + data[i - 1]["utterance"] + " [SEP] "
        else:
            context = ""
        if "sparql" not in answer:
            continue
        qa["qid"] = file_path.split("_")[-1][:-4] + str(i // 2)
        qa["question"] = context + question["utterance"] + " [CTX]"
        qa["sparql_query"] = answer["sparql"]
        try:
            qa["s_expression"] = parser.parse_query(answer["sparql"])
        except:
            print(qa["question"])
            print(qa["sparql_query"])
            print()
            continue
        entity_names = answer["utterance"].split(", ")
        qa["answer"] = [{"entity_name": entity_name} for entity_name in entity_names]
        processed_data.append(qa)
    return processed_data

def generate_test_data(file_path):
    data = json.load(open(file_path, 'r'), strict=False)
    processed_data = []
    for i in range(0, len(data), 2):
        question = data[i]
        answer = data[i + 1]
        qa = {}
        if not isdirect(question) and i > 0:
            context = context + data[i - 2]["utterance"] + " [SEP] " + data[i - 1]["utterance"] + " [SEP] "
        else:
            context = ""
        if "sparql" not in answer:
            continue
        qa["qid"] = i // 2
        qa["question"] = context + question["utterance"] + " [CTX]"
        processed_data.append(qa)
    return processed_data



root = "../data/SPICE/train/"
data = []
for dir in os.listdir(root):
    for file_name in os.listdir(root + dir):
        data += generate_train_data(root + dir + '/' + file_name)
json.dump(data, open("../data/processed_spice_data/train_full.json", 'w'), indent=2)

root = "../data/SPICE/valid/"
data = []
for dir in os.listdir(root):
    for file_name in os.listdir(root + dir):
        data += generate_test_data(root + dir + '/' + file_name)
json.dump(data, open("../data/processed_spice_data/dev_full.json", 'w'), indent=2)
# output_dir = "data/processed_spice_data/"
# shutil.rmtree(output_dir)
# os.mkdir(output_dir)

# dir = "data/SPICE/train/QA_0/"
# data = []
# for file_name in os.listdir(dir):
#     data += generate_train_data(dir + file_name)
# json.dump(data, open(output_dir + "train.json", 'w'), indent=4)
# dir = "data/SPICE/valid/QA_14/"
# data = []
# for file_name in os.listdir(dir):
#     data += generate_train_data(dir + file_name)
# json.dump(data, open(output_dir + "dev.json", 'w'), indent=4)