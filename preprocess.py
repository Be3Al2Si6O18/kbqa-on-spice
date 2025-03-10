import json
from utils.parse_expr import *
from utils.parse_sparql import *
data = json.load(open('data/processed_spice_data/train_full.json', 'r'))

sparql_parser = SparqlParser()
expr_parser = ExprParser()

for example in data:
    example['s_expression'] = sparql_parser.parse_query(example['sparql_query'])
    example['new_sparql'] = expr_parser.lisp_to_sparql(example['s_expression'])
json.dump(data, open('data/processed_spice_data/train_full.json', 'w'), indent=4)
