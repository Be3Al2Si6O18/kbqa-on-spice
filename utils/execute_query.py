from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://localhost:9999/blazegraph/sparql")
sparql.setReturnFormat(JSON)

def execute_query(query: str, multi_var=False):
    if not query: return []
    sparql.setQuery('prefix wd: <http://www.wikidata.org/entity/> prefix wdt: <http://www.wikidata.org/prop/direct/> ' + query)
    try:
        results = sparql.query().convert()
    except:
        print('SPARQL error:', query)
        return []
    if query.startswith('ASK'):
        return results['boolean']
    else:
        if not multi_var:
            return [binding[var]['value'].split('/')[-1] for binding in results['results']['bindings'] for var in binding]
        else:
            return [[binding[var]['value'].split('/')[-1] for var in binding] for binding in results['results']['bindings']]