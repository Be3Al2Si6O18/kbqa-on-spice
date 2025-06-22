import os
import json
from parse_sparql import *
import collections
import pickle

parser = SparqlParser()

def isdirect(question):
    if question["question-type"].startswith("Simple Question"):
        return question["question-type"][17:23] == "Direct"
    if "description" in question:
        if "ndirect" in question["description"]:
            return False
        if "ncomplete" in question["description"]:
            return False
    return True

def get_complete_question_type(qa):
    return f"{qa['question_type']} [{qa['description']}]" if 'description' else qa['question_type']

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

        dir_name, file_name = file_path.split('/')[-2:]
        qa['turnID'] = dir_name[3:] + '.' + file_name[3:-5] + '.' + str(i // 2)
        qa['question_type'] = question['question-type']
        qa['description'] = question.get('description', '')
        qa['question'] = context + question['utterance'] + " [CTX]"
        qa['answer'] = answer['utterance']
        qa['s_expression'] = parser.parse_sparql(answer['sparql'])
        qa['s_expression_cores'] = parser.s_expression_cores
        qa['sparql_delex'] = answer['sparql']
        qa['results'] = answer['all_entities']
        
        # qa["question_type"] = f"{question['question-type']} [{question['description']}]" if 'description' in question else question['question-type']
        
        # entity_names = answer["utterance"].split(", ")
        # qa["answer"] = [{"entity_name": entity_name} for entity_name in entity_names]
        processed_data.append(qa)
    return processed_data

# def generate_test_data(file_path):
#     data = json.load(open(file_path, 'r'), strict=False)
#     processed_data = []
#     for i in range(0, len(data), 2):
#         question = data[i]
#         answer = data[i + 1]
#         qa = {}
#         if not isdirect(question) and i > 0:
#             context = context + data[i - 2]["utterance"] + " [SEP] " + data[i - 1]["utterance"] + " [SEP] "
#         else:
#             context = ""
#         if "sparql" not in answer:
#             continue
#         qa["qid"] = i // 2
#         qa["question"] = context + question["utterance"] + " [CTX]"
#         processed_data.append(qa)
#     return processed_data

# root = "../data/SPICE/train/"
# data = []
# for dir in os.listdir(root):
#     for file_name in os.listdir(root + dir):
#         data += generate_train_data(root + dir + '/' + file_name)
# json.dump(data, open("../data/processed_spice_data/train_full.json", 'w'), indent=2)

# root = "../data/SPICE/valid/"
# data = []
# for dir in os.listdir(root):
#     for file_name in os.listdir(root + dir):
#         data += generate_train_data(root + dir + '/' + file_name)
#         if len(data) > 1000:
#             break
# json.dump(data, open("../data/processed_spice_data/dev_1000.json", 'w'), indent=2)

# each_type_num = 10

# root = "../data/SPICE/train/"
# data = collections.defaultdict(list)
# for dir in os.listdir(root):
#     for file_name in os.listdir(root + dir):
#         new_data =  generate_train_data(root + dir + '/' + file_name)

#         conversation = []
#         for example in new_data:
#             if "[SEP]" not in example['question']: # direct question
#                 if any([len(data[get_complete_question_type(qa)]) < each_type_num for qa in conversation]):
#                     for qa in conversation:
#                         data[get_complete_question_type(qa)].append(qa)
#                 conversation = [example]
#             else:
#                 conversation.append(example)

# json.dump([example for question_type in data for example in data[question_type]], open(f"../data/processed_spice_data/train_each_type_{each_type_num}.json", 'w'), indent=2)
# for question_type in data:
#     print(question_type, len(data[question_type]))

each_type_num = 50

root = "../data/SPICE/valid/"
num_question_each_type = collections.defaultdict(int)
selected_data = []
for dir in os.listdir(root):
    for file_name in os.listdir(root + dir):
        new_data =  generate_train_data(root + dir + '/' + file_name)

        conversation = []
        for example in new_data:
            if "[SEP]" not in example['question']: # new conversation
                if any([num_question_each_type[get_complete_question_type(qa)] < each_type_num for qa in conversation]):
                    for qa in conversation:
                        num_question_each_type[get_complete_question_type(qa)] += 1
                        selected_data.append(qa)
                conversation = [example]
            else:
                conversation.append(example)

exp_name = f'dev_each_type_{each_type_num}'
# output_path = f'../output/{exp_name}/'
# os.makedirs(output_path, exist_ok=True)

json.dump(selected_data, open(f"../data/processed_spice_data/dev_each_type_{each_type_num}.json", 'w'), indent=2)

# with open(output_path + 'each_type_count.txt', 'w') as f:
#     for question_type in data:
#         print(question_type, len(data[question_type]), file=f)