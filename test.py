import json
import urllib
import time
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://localhost:9999/blazegraph/sparql")
sparql.setReturnFormat(JSON)

def execute_query(query: str):
    sparql.setQuery('prefix wd: <http://www.wikidata.org/entity/> prefix wdt: <http://www.wikidata.org/prop/direct/> ' + query)
    try:
        results = sparql.query().convert()
    except:
    # except urllib.error.URLError:
        print(query)
        exit(0)
    if query.startswith('ASK'):
        return results['boolean']
    else:
        return [binding[var]['value'].split('/')[-1] for binding in results['results']['bindings'] for var in binding]

data = json.load(open('data/processed_spice_data/train_full.json', 'r'))[:10000]

correct = 0
total = 0
t0 = time.time()
for i in range(len(data)):
    time.sleep(0.1)
    example = data[i]
    if i % 100 == 0 and i > 0:
        t = time.time()
        print(f"{i}/{len(data)} -- {correct}/{total} -- {t - t0:.2f}s", flush=True)
    results0 = execute_query(example['sparql_query'])
    results = execute_query(example['new_sparql'])
    if results0 == results or set(results0) == set(results):
        correct += 1
    else:
        print(example['sparql_query'])
        print()
        print(example['s_expression'])
        print()
        print(example['new_sparql'])
        if len(results0) > 1:
            print(len(results0), len(results))
        else:
            print(results0[0], results[0])
    total += 1

print(correct, total)
print(f"time: {(time.time() - t0):.2f}s")