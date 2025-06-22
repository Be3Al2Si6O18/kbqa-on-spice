import json
from utils.execute_query import execute_query

data = json.load(open('output/memory_core_1/prediction.json', 'r'))

new_data = []
for i, qa in enumerate(data):
    results1 = execute_query(qa['sparql_delex'])
    results2 = execute_query(qa['actions'])
    if isinstance(results1, bool):
        if results1 != results2:
            new_data.append(qa)
    elif isinstance(results1, list):
        if not isinstance(results2, list) or sorted(results1) != sorted(results2):
            new_data.append(qa)
    if (i + 1) % 100 == 0:
        print(f'{i + 1}/{len(data)}', flush=True)
json.dump(new_data, open('prediction.json', 'w'), indent=2)

attr_list = ['turnID', 'question_type', 'question', 'coreference_resolved_question', 's_expression_cores_fn', 'predicted_cores', 'calibrated_cores_fn', 'simple_question_type', 'predicted_simple_question_type', 'template', 'predicted_template', 'replacements', 'predicted_replacements', 'sparql_attempt_count', 's_expression_fn', 'predicted_s_expression_fn', 'sparql_delex', 'actions']
simple_data = [{attr: qa[attr] for attr in attr_list} for qa in new_data]
json.dump(simple_data, open('simple_prediction.json', 'w'), indent=2)