"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

class ParseError(Exception):
    pass

class Parser:
    def __init__(self):
        pass
    
    def parse_query(self, query):
        if query.startswith('ASK'):
            body = query[3:]
            body_lines = body.strip('} {').split('.')[:-1]
            triplets = ['(' + ' '.join([self.remove_prefix(elem) for elem in body_line.strip().split(' ')]) + ')' for body_line in body_lines]
            return '(ASK {})'.format(' '.join(triplets))

        if query.startswith('SELECT'):
            query = query[6:]
        else:
            raise ParseError()
        query = query.lstrip()

        ret_var = '?x'
        decorators = []
        if query.startswith('?x'):
            query = query[2:]
        elif query.startswith('?y'):
            query = query[2:]
            ret_var = '?y'
        elif query.startswith('DISTINCT ?x'):
            query = query[11:]
            decorators.append('DISTINCT')
        elif query.startswith('(COUNT(*) AS ?count)'):
            query = query[20:]
            decorators.append('COUNT')
        elif query.startswith('(COUNT(*) AS ?result)'):
            query = query[22:]
            decorators.append('COUNT')
        elif query.startswith('(COUNT(DISTINCT ?x) AS ?count)'):
            query = query[30:]
            decorators.append('COUNT')
            decorators.append('DISTINCT')
        query = query.lstrip()

        if query.startswith('WITH'):
            subqueries = []
            while query.startswith('WITH'):
                begin = 6
                stack = 1
                i = begin
                while i < len(query):
                    if query[i] == '{':
                        stack += 1
                    elif query[i] == '}':
                        stack -= 1
                        if stack == 0:
                            end = i
                            break
                    i += 1
                subquery = query[begin:end].lstrip()
                query = query[end + 1:].lstrip()[2:].lstrip().split(' ', 1)[1].lstrip()
                subqueries.append(subquery)
            s_expr = self.parse_subqueries(subqueries, query)
        elif query.startswith('WHERE'):
            body = query[6:]
            if 'UNION' not in body:
                s_expr = self.parse_naive_body(body, ret_var)
            else:
                s_expr = '(OR ' + ' '.join([(self.parse_naive_body(body, ret_var)) for body in body.split(' UNION ')]) + ')'
        else:
            raise ParseError()

        for decorator in decorators[::-1]:
            s_expr = '({} {})'.format(decorator, s_expr)
        
        return s_expr
        
    def parse_subqueries(self, subqueries, filter):
        tupcountfin = self.parse_subquery(subqueries[0])
        if len(subqueries) > 4:
            tupcounts2 = self.parse_subquery(subqueries[2])
            tupcountfin = '(GROUP_SUM {} {})'.format(tupcountfin, tupcounts2)

        if len(subqueries) == 4 or len(subqueries) == 6:
            if 'MIN' in subqueries[-1]:
                return '(ARGMIN {})'.format(tupcountfin)
            elif 'MAX' in subqueries[-1]:
                return '(ARGMAX {})'.format(tupcountfin)
            else:
                raise ParseError()
        
        operator_to_function = {'<': 'LT', '<=': 'LE', '=': 'EQ', '>=': 'GE', '>': 'GT'}
        pattern = r'FILTER \((.*?) (.*?) (.*?)\)'
        re_match = re.search(pattern, filter)
        if re_match:
            operator = re_match.group(2)
            if len(subqueries) == 3 or len(subqueries) == 5:
                num = re_match.group(3)
            elif len(subqueries) == 2:
                num_body = re.search(r'SELECT \(COUNT\(\*\) AS \?count\) WHERE {(.*?)}', filter).group(1)
                num = self.parse_naive_body(num_body, '?w')
            else:
                raise ParseError()
            return '({} {} {})'.format(operator_to_function[operator], tupcountfin, num)
        else:
            return tupcountfin
    
    def parse_subquery(self, subquery):
        self.parse_assert(subquery.startswith('SELECT'))
        subquery = subquery[6:]
        subquery = subquery.lstrip()
        
        if subquery.startswith('?x'):
            subquery = subquery[2:]
            ret_var = '?x'
        elif subquery.startswith('?y'):
            subquery = subquery[2:]
            ret_var = '?y'
        else:
            raise ParseError()
        subquery = subquery.lstrip()

        self.parse_assert(subquery.startswith('(COUNT(*) AS ?tupcount)'))
        subquery = subquery[23:]
        subquery = subquery.lstrip()

        self.parse_assert(subquery.startswith('WHERE'))
        body = subquery[6:]
        s_expr = '(GROUP_COUNT {})'.format(self.parse_naive_body(body, ret_var))
        return s_expr
    
    def remove_prefix(self, elem):
        if elem.startswith('wd:'):
            return elem[3:]
        elif elem.startswith('wdt:'):
            return elem[4:]
        else:
            return elem

    def parse_naive_body(self, body, ret_var):
        body_lines = body.strip('} {').split('.')[:-1]

        def split_body_line(body_line):
            body_line = body_line.strip()
            if body_line.startswith('VALUES ?y'):
                values = ' '.join([self.remove_prefix(value) for value in body_line[12:-2].split(' ')])
                triplet = ['?y', 'VALUES', values]
            else:
                triplet = [self.remove_prefix(elem) for elem in body_line.split(' ')]
            return triplet
        
        diff = None
        if body_lines[-1].lstrip().startswith('FILTER NOT EXISTS'):
            triplet = split_body_line(re.search(r'FILTER NOT EXISTS {(.*)', body_lines.pop()).group(1))
            diff = self.triplet_to_clause('?x', triplet, {})
            
        triplets = [split_body_line(body_line) for body_line in body_lines]
        
        triplets_pool = triplets
        var_dep_list = []
        successors = []
        dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))   

        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, tgt_var, successors)
            self.parse_assert(len(dep_triplets) > 0)
            var_dep_list.append((tgt_var, dep_triplets))
        
        self.parse_assert(len(triplets_pool) == 0)

        s_expr = self.dep_graph_to_s_expr(var_dep_list, ret_var)
        if diff:
            s_expr = '(DIFF {} {})'.format(s_expr, diff)

        return s_expr
    
    def resolve_dependancy(self, triplets, target_var, successors):
        dep = []
        left = []
        for tri in triplets:
            if tri[0] == target_var:
                dep.append(tri)
                if tri[-1].startswith('?') and tri[-1] not in successors:
                    successors.append(tri[-1])
            elif tri[-1] == target_var:
                dep.append(tri)
                if tri[0].startswith('?') and tri[0] not in successors:
                    successors.append(tri[0])
            else:
                left.append(tri)
        return dep, left
    
    def dep_graph_to_s_expr(self, var_dep_list, ret_var, spec_condition=None):
        self.parse_assert(var_dep_list[0][0] == ret_var)
        var_dep_list.reverse()
        parsed_dict = {}

        for var_name, dep_relations in var_dep_list:
            # clause = self.triplet_to_clause(var_name, dep_relations[0], parsed_dict)
            # for tri in dep_relations[1:]:
            #     n_clause = self.triplet_to_clause(var_name, tri, parsed_dict)
            #     clause = '(AND {} {})'.format(clause, n_clause)
            if len(dep_relations) == 1:
                clause = self.triplet_to_clause(var_name, dep_relations[0], parsed_dict)
            else:
                clause = '(AND ' + ' '.join([self.triplet_to_clause(var_name, tri, parsed_dict) for tri in dep_relations]) + ')'
            parsed_dict[var_name] = clause
        return parsed_dict[ret_var]
    
    def triplet_to_clause(self, tgt_var, triplet, parsed_dict):
        if triplet[1] == 'VALUES':
            return '(VALUES {})'.format(triplet[-1])
        if triplet[0] == tgt_var:
            other = triplet[-1]
            if other in parsed_dict:
                other = parsed_dict[other]
            return '(JOIN {} {})'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            other = triplet[0]
            if other in parsed_dict:
                other = parsed_dict[other]
            return '(JOIN (R {}) {})'.format(triplet[1], other)
        else:
            raise ParseError()
    
    def parse_assert(self, eval):
        if not eval:
            raise ParseError()

if __name__ == '__main__':
    parser = Parser()
    examples  = [
        'SELECT ?x WHERE { wd:Q35 wdt:P361 ?x . ?x wdt:P31 wd:Q2221906 .  }',
        'SELECT ?x WHERE { { wd:Q12060361 wdt:P17 ?x . ?x wdt:P31 wd:Q1048835 .  } UNION { wd:Q5394403 wdt:P27 ?x . ?x wdt:P31 wd:Q1048835 .  } }',
        'SELECT (COUNT(*) AS ?count) WHERE { wd:Q172771 wdt:P17 ?x . ?x wdt:P31 wd:Q15617994 .  }',
        'SELECT (COUNT(DISTINCT ?x) AS ?count) WHERE { { wd:Q26208739 wdt:P136 ?x . ?x wdt:P31 wd:Q151885 .  } UNION { wd:Q5323744 wdt:P136 ?x . ?x wdt:P31 wd:Q151885 .  } }',
        'SELECT DISTINCT ?x WHERE { ?x wdt:P355 ?y . VALUES ?y { wd:Q7163209 wd:Q5908783 wd:Q1273145 }. ?x wdt:P31 wd:Q43229 .  }',
        'SELECT ?x WHERE { ?x wdt:P710 wd:Q526275 . ?x wdt:P31 wd:Q1203472 .  FILTER NOT EXISTS { ?x wdt:P710 wd:Q1736517 .  } }',
        'ASK { wd:Q230 wdt:P530 wd:Q35 .  wd:Q230 wdt:P530 wd:Q40 .  wd:Q230 wdt:P530 wd:Q664 .  }',
        'SELECT ?y  WITH { SELECT ?y (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P495 ?y . ?x wdt:P31 wd:Q2031291 . ?y wdt:P31 wd:Q15617994 .  } GROUP BY ?y } AS  %tupcounts WITH {  SELECT DISTINCT ?y (0 AS ?tupcount) WHERE {  { { ?b wdt:P495 ?y . ?y wdt:P31 wd:Q15617994 .  } }  FILTER NOT EXISTS { ?x wdt:P495 ?y . ?x wdt:P31 wd:Q2031291 . ?y wdt:P31 wd:Q15617994 .  } }  } AS %zerotupcounts  WHERE { { SELECT ?y ?tupcount WHERE { INCLUDE %tupcounts } }  UNION { SELECT ?y ?tupcount WHERE { INCLUDE %zerotupcounts } }  FILTER (?tupcount > ?count){SELECT (COUNT(*) AS ?count) WHERE {wd:Q717 wdt:P495 ?w . ?w wdt:P31 wd:Q15617994 . }} }',
        'SELECT (COUNT(*) AS ?result) WITH { SELECT  ?x (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P530 ?y . ?x wdt:P31 wd:Q15617994 . ?y wdt:P31 wd:Q15617994 .  } GROUP BY ?x }  AS %tupcounts  WITH { SELECT DISTINCT ?x (0 AS ?tupcount) WHERE { { { ?x wdt:P530 ?b . ?x wdt:P31 wd:Q15617994 .  } } FILTER NOT EXISTS { ?x wdt:P530 ?y . ?x wdt:P31 wd:Q15617994 . ?y wdt:P31 wd:Q15617994 .  } } } AS %zerotupcounts  WITH { SELECT ?x ?tupcount WHERE { { SELECT ?x ?tupcount WHERE { INCLUDE %tupcounts } } UNION { SELECT ?x ?tupcount WHERE { INCLUDE %zerotupcounts } } } } AS %TuplesCounts  WHERE  { INCLUDE %TuplesCounts . FILTER (?tupcount > 0) }',
        'SELECT  ?x  WITH { SELECT  ?x (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q3895768 . ?y wdt:P31 wd:Q2342494 .  } GROUP BY ?x }  AS %tupcounts  WITH { SELECT DISTINCT ?x (0 AS ?tupcount) WHERE { { { ?x wdt:P1441 ?b . ?x wdt:P31 wd:Q3895768 .  } } FILTER NOT EXISTS { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q3895768 . ?y wdt:P31 wd:Q2342494 .  } } } AS %zerotupcounts  WITH { SELECT ?x ?tupcount WHERE { { SELECT ?x ?tupcount WHERE { INCLUDE %tupcounts } } UNION { SELECT ?x ?tupcount WHERE { INCLUDE %zerotupcounts } } } } AS %TuplesCounts  WITH { SELECT (MAX(?tupcount) AS ?count) WHERE { INCLUDE %TuplesCounts } } AS %maxMinCount  WHERE { INCLUDE %TuplesCounts . INCLUDE %maxMinCount .  FILTER (?tupcount = ?count) } ',
        'SELECT (COUNT(*) AS ?result) WITH { SELECT ?x (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q502895 . ?y wdt:P31 wd:Q15416 .  } GROUP BY ?x }  AS %tupcounts1  WITH { SELECT DISTINCT ?x (0 AS ?tupcount) WHERE { { { ?x wdt:P1441 ?b . ?x wdt:P31 wd:Q502895 .  } } FILTER NOT EXISTS { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q502895 . ?y wdt:P31 wd:Q15416 .  } } } AS %zerotupcounts1  WITH { SELECT ?x (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q502895 . ?y wdt:P31 wd:Q838948 .  } GROUP BY ?x }  AS %tupcounts2 WITH { SELECT DISTINCT ?x (0 AS ?tupcount) WHERE { { { ?x wdt:P1441 ?b . ?x wdt:P31 wd:Q502895 .  } }  FILTER NOT EXISTS { ?x wdt:P1441 ?y . ?x wdt:P31 wd:Q502895 . ?y wdt:P31 wd:Q838948 .  } } } AS %zerotupcounts2 WITH {SELECT ?x (SUM(?tupcount) AS ?tupcountfin) WHERE { { SELECT ?x ?tupcount WHERE { INCLUDE %tupcounts1 } } UNION { SELECT ?x ?tupcount WHERE { INCLUDE %zerotupcounts1 } } UNION { SELECT ?x ?tupcount WHERE { INCLUDE %tupcounts2 } } UNION { SELECT ?x ?tupcount WHERE { INCLUDE %zerotupcounts2 } } } GROUP BY ?x } AS %TuplesCounts  WHERE {INCLUDE %TuplesCounts. FILTER (?tupcountfin > 0) }',
        'SELECT ?y WITH { SELECT ?y (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 . ?x wdt:P31 wd:Q13406554 .  } GROUP BY ?y }  AS %tupcounts1  WITH { SELECT DISTINCT ?y (0 AS ?tupcount) WHERE { { { ?b wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 .  } } FILTER NOT EXISTS { ?x wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 . ?x wdt:P31 wd:Q13406554 .  } } } AS %zerotupcounts1  WITH { SELECT ?y (COUNT(*) AS ?tupcount) WHERE { ?x wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 . ?x wdt:P31 wd:Q15275719 .  } GROUP BY ?y }  AS %tupcounts2 WITH { SELECT DISTINCT ?y (0 AS ?tupcount) WHERE { { { ?b wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 .  } }  FILTER NOT EXISTS { ?x wdt:P1923 ?y . ?y wdt:P31 wd:Q1194951 . ?x wdt:P31 wd:Q15275719 .  } } } AS %zerotupcounts2  WITH {  SELECT ?y (SUM(?tupcount) AS ?tupcountfin) WHERE { { SELECT ?y ?tupcount WHERE { INCLUDE %tupcounts1 } } UNION { SELECT ?y ?tupcount WHERE { INCLUDE %zerotupcounts1 } } UNION { SELECT ?y ?tupcount WHERE { INCLUDE %tupcounts2 } } UNION { SELECT ?y ?tupcount WHERE { INCLUDE %zerotupcounts2 } }  } GROUP BY ?y } AS %TuplesCounts  WITH { SELECT (MIN(?tupcountfin) AS ?count) WHERE { INCLUDE %TuplesCounts } } AS %maxMinCount  WHERE { INCLUDE %TuplesCounts . INCLUDE %maxMinCount .  FILTER (?tupcountfin = ?count) }'
    ]
    for query in examples:
        print('--------------------------------')
        print()
        print(query)
        print()
        print(parser.parse_query(query))
        print()