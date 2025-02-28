from typing import List
import copy

class ParseError(Exception):
    pass

def lisp_to_sparql(lisp_program: str):
    nested_expression = lisp_to_nested_expression(lisp_program)
    return nested_expression_to_sparql(nested_expression)

def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def nested_expression_to_sparql(expression):
    # print(expression)

    select = 'SELECT ?x '
    if expression[0] == 'COUNT':
        select = select.replace('?x', 'COUNT(?x)')
        expression = expression[1]
    if expression[0] == 'DISTINCT':
        select = select.replace('?x', 'DINSTINCT ?x')
        expression = expression[1]

    if expression[0] in ['JOIN', 'AND', 'OR', 'DIFF']:
        body = generate_body(expression)
        return select + 'WHERE { ' + body + ' }'
    
    elif expression[0] in ['LT', 'LE', 'EQ', 'GE', 'GT']:
        function_to_operator = {'LT': '<', 'LE': '<=', 'EQ': '=', 'GE': '>=', 'GT': '>'}
        operator = function_to_operator[expression[0]]
        subqueries = generate_subqueries(expression[1])
        if isinstance(expression[2], list):
            count_body = generate_body(expression[2])
            subqueries += ' WITH { SELECT (COUNT(*) AS ?count) WHERE { ' + count_body + ' } } AS %Count'
            return select + subqueries + ' WHERE { INCLUDE %TuplesCounts . INCLUDE %Count . FILTER (?tupcountfin ' + operator + ' ?count) }'
        else:
            return select + subqueries + ' WHERE { INCLUDE %TuplesCounts . FILTER (?tupcountfin ' + operator + ' ' + expression[2] + ') }'
    
    elif expression[0] in ['ARGMAX', 'ARGMIN']:
        arg = 'MAX' if expression[0] == 'ARGMAX' else 'MIN'
        subqueries = generate_subqueries(expression[1], arg)
        return select + subqueries + ' WHERE { INCLUDE %TuplesCounts . INCLUDE %maxMinCount . FILTER (?tupcountfin = ?count) }'
    
    elif expression[0] == 'ASK':
        return 'ASK { ' + ' '.join([f'wd:{triplets[0]} wdt:{triplets[1]} wd:{triplets[2]} .' for triplets in expression[1:]]) + ' }'
    
    else:
        raise ParseError()

def generate_body(expression, zerotupcounts=False):

    def linearize_lisp_expression(expression: list, sub_formula_id):
        expression = copy.deepcopy(expression)
        sub_formulas = []
        for i, e in enumerate(expression):
            if isinstance(e, list) and e[0] != 'R':
                sub_formulas.extend(linearize_lisp_expression(e, sub_formula_id))
                expression[i] = '#' + str(sub_formula_id[0] - 1)

        sub_formulas.append(expression)
        sub_formula_id[0] += 1
        return sub_formulas
    
    if expression[0] == 'DIFF':
        body1, body2 = generate_body(expression[1]), generate_body(expression[2])
        return body1 + ' FILTER NOT EXISTS { ' + body2 + ' }'
    
    elif expression[0] == 'OR':
        return '{ '+ ' } UNION { '.join([generate_body(sub_expression) for sub_expression in expression[1:]]) + ' }'
    
    elif expression[0] in ['JOIN', 'AND']:
        sub_programs = linearize_lisp_expression(expression, [0])

        clauses = []
        identical_variables_r = {}  # key should be larger than value
        question_var = len(sub_programs) - 1

        def get_root(var: int):
            while var in identical_variables_r:
                var = identical_variables_r[var]
            return var

        for i, subp in enumerate(sub_programs):
            i = str(i)
            if subp[0] == 'JOIN':
                if isinstance(subp[1], list):  # R relation
                    if subp[2][0] in ["P", "Q"]:  # entity
                        clauses.append("wd:" + subp[2] + " wdt:" + subp[1][1] + " ?x" + i + " .")
                    elif subp[2][0] == '#':  # variable
                        clauses.append("?x" + subp[2][1:] + " wdt:" + subp[1][1] + " ?x" + i + " .")
                    else:
                        raise ParseError()
                else:
                    if subp[2][0] in ["P", "Q"]:  # entity
                        clauses.append("?x" + i + " wdt:" + subp[1] + " wd:" + subp[2] + " .")

                    elif subp[2][0] == '#':  # variable
                        clauses.append("?x" + i + " wdt:" + subp[1] + " ?x" + subp[2][1:] + " .")
                    else:
                        raise ParseError()
            elif subp[0] == 'AND':
                root_this = get_root(int(i))
                for var in subp[1:]:
                    parse_assert(var[0] == "#")
                    root_var = get_root(int(var[1:]))
                    if root_this > root_var:
                        identical_variables_r[root_this] = root_var
                        root_this = root_var
                    else:
                        identical_variables_r[root_var] = root_this
            elif subp[0] == 'VALUES':
                clauses.append('VALUES ?x' + i + ' { ' + ' '.join([f'wd:{value}' for value in subp[1:]]) + ' } .')
            else:
                raise ParseError()
        
        question_var = get_root(question_var)

        for i in range(len(clauses)):
            for k in identical_variables_r:
                clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')
        
        if zerotupcounts:
            for i in range(len(clauses)):
                for j in range(len(sub_programs)):
                    if j != question_var:
                        clauses[i] = clauses[i].replace(f'?x{j} ', f'?b{j} ')
            clauses = [clauses[i] for i in range(len(clauses)) if '?x ' in clauses[i]]
        
        return ' '.join(clauses)
    else:
        raise ParseError()

def generate_subqueries(expression, arg=None):

    def generate_tupcounts(expression):
        parse_assert(expression[0] == 'GROUP_COUNT')
        tupcounts_body = generate_body(expression[1])
        zerotupcounts_body = generate_body(expression[1], zerotupcounts=True)
        tupcounts = 'WITH { SELECT ?x (COUNT(*) AS ?tupcount) WHERE { ' + tupcounts_body + ' } GROUP BY ?x } AS %tupcounts'
        zerotupcounts = 'WITH { SELECT DISTINCT ?x (0 AS ?tupcount) WHERE { ' + zerotupcounts_body + ' FILTER NOT EXISTS { ' + tupcounts_body + ' } } } AS %zerotupcounts'
        return [tupcounts, zerotupcounts]

    def generate_TuplesCounts(tupcounts_dict, sum=False):
        body = ' UNION '.join(['{ SELECT ?x ?tupcount WHERE { INCLUDE %' + tupcounts.split('%')[1] + ' } }' for tupcounts in tupcounts_dict])
        if sum:
            return 'WITH { SELECT ?x (SUM(?tupcount) AS ?tupcountfin) WHERE { ' + body + ' } GROUP BY ?x } AS %TuplesCounts'
        else:
            return 'WITH { SELECT ?x (?tupcount AS ?tupcountfin) WHERE { ' + body + ' } } AS %TuplesCounts'
    
    def generate_maxMinCount(arg):
        return 'WITH { SELECT (' + arg + '(?tupcountfin) AS ?count) WHERE { INCLUDE %TuplesCounts } } AS %maxMinCount'
    
    subqueries = []

    if expression[0] == 'GROUP_COUNT':
        subqueries = generate_tupcounts(expression)
        subqueries.append(generate_TuplesCounts(subqueries))

    elif expression[0] == 'GROUP_SUM':
        tupcounts1, zerotupcounts1 = generate_tupcounts(expression[1])
        tupcounts2, zerotupcounts2 = generate_tupcounts(expression[2])
        subqueries = [tupcounts1 + '1', zerotupcounts1 + '1', tupcounts2 + '2', zerotupcounts2 + '2']
        subqueries.append(generate_TuplesCounts(subqueries, sum=True))
    else:
        raise ParseError()

    if arg:
        subqueries.append(generate_maxMinCount(arg))

    return ' '.join(subqueries)


def parse_assert(eval):
    if not eval:
        raise ParseError()

if __name__ == '__main__':
    expressions = [
        '(AND (JOIN (R P361) Q35) (JOIN P31 Q2221906))',
        '(OR (AND (JOIN (R P17) Q12060361) (JOIN P31 Q1048835)) (AND (JOIN (R P27) Q5394403) (JOIN P31 Q1048835)))',
        '(COUNT (AND (JOIN (R P17) Q172771) (JOIN P31 Q15617994)))',
        '(COUNT (DISTINCT (OR (AND (JOIN (R P136) Q26208739) (JOIN P31 Q151885)) (AND (JOIN (R P136) Q5323744) (JOIN P31 Q151885)))))',
        '(DISTINCT (AND (JOIN P355 (VALUES Q7163209 Q5908783 Q1273145)) (JOIN P31 Q43229)))',
        '(DIFF (AND (JOIN P710 Q526275) (JOIN P31 Q1203472)) (JOIN P710 Q1736517))',
        '(ASK (Q230 P530 Q35) (Q230 P530 Q40) (Q230 P530 Q664))',
        '(GT (GROUP_COUNT (AND (JOIN (R P495) (JOIN P31 Q2031291)) (JOIN P31 Q15617994))) (AND (JOIN (R P495) Q717) (JOIN P31 Q15617994)))',
        '(COUNT (GT (GROUP_COUNT (AND (JOIN P530 (JOIN P31 Q15617994)) (JOIN P31 Q15617994))) 0))',
        '(ARGMAX (GROUP_COUNT (AND (JOIN P1441 (JOIN P31 Q2342494)) (JOIN P31 Q3895768))))',
        '(COUNT (GT (GROUP_SUM (GROUP_COUNT (AND (JOIN P1441 (JOIN P31 Q15416)) (JOIN P31 Q502895))) (GROUP_COUNT (AND (JOIN P1441 (JOIN P31 Q838948)) (JOIN P31 Q502895)))) 0))',
        '(ARGMIN (GROUP_SUM (GROUP_COUNT (AND (JOIN (R P1923) (JOIN P31 Q13406554)) (JOIN P31 Q1194951))) (GROUP_COUNT (AND (JOIN (R P1923) (JOIN P31 Q15275719)) (JOIN P31 Q1194951)))))'
    ]

    for expression in expressions:
        print('--------------------------------')
        print()
        print(expression)
        print()
        print(lisp_to_sparql(expression))
        print()