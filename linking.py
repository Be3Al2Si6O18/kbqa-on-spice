import json
import numpy as np
import itertools
from retriever.semantic_retriever import SemanticRetriever
from utils.execute_query import execute_query
from utils.parse_expr import expression_to_sparql

entity_retriever = SemanticRetriever('entity')
relation_retriever = SemanticRetriever('relation')
rela_mid_to_faiss_index = {mid: i for i, mid in enumerate(relation_retriever.mid_list)}
rela_fn_to_faiss_index = {fn: i for i, fn in enumerate(relation_retriever.fn_list)}
wikidata_mid_to_fn = json.load(open('data/wikidata_mid_to_fn.json', 'r'))

def is_close(expr):
    stack = 0
    for char in expr:
        if char == '(':
            stack += 1
        elif char == ')':
            if stack == 0:
                return False
            stack -= 1
    return stack == 0

def fix_core(core):
    tokens = [token.replace('(', '').replace(')', '') for token in core.split()]
    index = 0
    def parse_core():
        nonlocal index
        func_list = ['JOIN', 'R', 'AND', 'VALUES', 'IS_TRUE']
        token = tokens[index]
        if token in func_list:
            index += 1
            args = []
            if token == 'IS_TRUE':
                for _ in range(3):
                    args.append(parse_core())
            elif token == 'JOIN':
                for _ in range(2):
                    args.append(parse_core())
            elif token == 'R':
                args.append(parse_core())
            elif token == 'AND':
                while index < len(tokens):
                    args.append(parse_core())
            else:
                while index < len(tokens) and tokens[index] not in func_list:
                    args.append(parse_core())
            return f'({token} {' '.join(args)})'
        else:
            value = token
            index += 1
            return value
    try:
        fixed_core = parse_core()
    except:
        fixed_core = core
    return fixed_core

def get_1hop_relations(entity):
    query = "SELECT DISTINCT ?x0 WHERE { { ?x1 ?x0 wd:" + entity + " . } UNION { wd:" + entity + " ?x0 ?x1 . } }"
    return [rela for rela in execute_query(query) if rela[0] == 'P']

def calculate_rela_similarity(vec1, rela_mid):
    def get_vector_from_mid(rela):
        if rela not in relation_retriever.mid_list:
            rela = relation_retriever.semantic_search(wikidata_mid_to_fn[rela])[0][1]
        return relation_retriever.index.reconstruct(rela_mid_to_faiss_index[rela])

    vec2 = get_vector_from_mid(rela_mid)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def add_reverse(org_exp):
    final_candi = [org_exp]
    total_join = 0
    list_seg = org_exp.split(" ")
    for seg in list_seg:
        if "JOIN" in seg:
            total_join += 1
    for i in range(total_join):
        final_candi = final_candi + add_reverse_index(final_candi, i + 1)
    return final_candi

def add_reverse_index(list_of_e, join_id):
    added_list = []
    list_of_e_copy = list_of_e.copy()
    for exp in list_of_e_copy:
        list_seg = exp.split(" ")
        count = 0
        for i, seg in enumerate(list_seg):
            if "JOIN" in seg and list_seg[i + 1] != "(R":
                count += 1
                if count != join_id:
                    continue
                if list_seg[i + 1] == 'P31':
                    break
                list_seg[i + 1] = "(R " + list_seg[i + 1] + ")"
                added_list.append(" ".join(list_seg))
                break
            if "JOIN" in seg and list_seg[i + 1] == "(R":
                count += 1
                if count != join_id:
                    continue
                list_seg[i + 1] = ""
                list_seg[i + 2] = list_seg[i + 2][:-1]
                added_list.append(" ".join(" ".join(list_seg).split()))
                break
    return added_list

def bound_to_existed(s_expression):
    query_count = 0
    try:
        expression_segment = s_expression.split(" ")

        type_count = 0
        type_replace_dict = {}
        for i, seg in enumerate(expression_segment):
            processed_seg = seg.strip(')')
            if i > 0 and expression_segment[i - 1] == 'instance_of':
                type_count += 1
                type_replace_dict[i] = f'?t{type_count}' + ')' * (len(seg) - len(processed_seg))
        for i in type_replace_dict:
            expression_segment[i - 1] = 'P31'
            expression_segment[i] = type_replace_dict[i]

        print(' '.join(expression_segment))

        enti_replace_dict = {}
        for i, seg in enumerate(expression_segment):
            processed_seg = seg.strip(')')
            if processed_seg[0] != '(' and not processed_seg.isdigit() and not (i > 0 and expression_segment[i - 1] in ['(R', '(JOIN'] or i > 1 and expression_segment[i - 2] == '(IS_TRUE') and not processed_seg.startswith('?t'):
                enti_replace_dict[i] = processed_seg
        if len(enti_replace_dict) > 4:
            top_k = 1
        elif len(enti_replace_dict) > 2:
            top_k = 2
        else:
            top_k = 3

        for i in enti_replace_dict:
            enti_replace_dict[i] = [mid for score, mid, fn in entity_retriever.semantic_search(enti_replace_dict[i], top_k)]

        print(enti_replace_dict)
        
        combinations = list(enti_replace_dict.values())
        all_iters = list(itertools.product(*combinations)) # 所有可能的实体替换方案
        enti_index = list(enti_replace_dict.keys()) # 待替换实体在expression_segment中的index
        for iters in all_iters:
            expression_segment_copy = expression_segment.copy()
            for i in range(len(iters)):
                cur_enti = expression_segment[enti_index[i]]
                suffix = ')' * cur_enti.count(')')
                expression_segment_copy[enti_index[i]] = iters[i] + suffix

            rela_replace_dict = {}
            for j, seg in enumerate(expression_segment):
                processed_seg = seg.strip(')')
                if processed_seg[0] != '(' and not processed_seg.isdigit() and (j > 0 and expression_segment[j - 1] in ['(R', '(JOIN'] or j > 1 and expression_segment[j - 2] == '(IS_TRUE') and not processed_seg == 'P31':
                    if expression_segment[j + 1] == '(JOIN':
                        rela_replace_dict[j] = [mid for score, mid, fn in relation_retriever.semantic_search(processed_seg, 4)]
                    else:
                        possible_rela = []
                        if expression_segment[j + 1] == '(VALUES':
                            idx = j + 2
                            while idx < len(expression_segment) and expression_segment[idx][0] != '(':
                                possible_rela += get_1hop_relations(expression_segment_copy[idx].strip(')'), )
                                idx += 1
                        else:
                            possible_rela += get_1hop_relations(expression_segment_copy[j + 1].strip(')'))
                        possible_rela = set(possible_rela)
                        seg_fn = relation_retriever.semantic_search(processed_seg)[0][2] if processed_seg not in relation_retriever.fn_list else processed_seg
                        seg_vector = relation_retriever.index.reconstruct(rela_fn_to_faiss_index[seg_fn])
                        rela_similarity = [(rela, calculate_rela_similarity(seg_vector, rela)) for rela in possible_rela if rela in wikidata_mid_to_fn]
                        rela_similarity.sort(key=lambda x: x[1], reverse=True)
                        rela_replace_dict[j] = [rela for rela, score in rela_similarity]
            if len(rela_replace_dict) > 4:
                top_k = 1
            elif len(rela_replace_dict) > 2:
                top_k = 2
            else:
                top_k = 4

            for j in rela_replace_dict:
                rela_replace_dict[j] = rela_replace_dict[j][:top_k]

            combinations_rela = list(rela_replace_dict.values())
            all_iters_rela = list(itertools.product(*combinations_rela))
            rela_index = list(rela_replace_dict.keys())
            for iter_rela in all_iters_rela:
                for k in range(len(iter_rela)):
                    cur_rela = expression_segment[rela_index[k]]
                    suffix = ')' * cur_rela.count(')')
                    expression_segment_copy[rela_index[k]] = iter_rela[k] + suffix
                final = " ".join(expression_segment_copy)
                added = add_reverse(final) # 反转关系生成变体
                # 遍历added，首次能够查询到结果时，返回查询结果
                for exp in added:
                    sparql = expression_to_sparql(exp)
                    answer = execute_query(sparql)
                    query_count += 1
                    if isinstance(answer, bool) or (answer and not (answer[0].isnumeric() and answer[0] == '0')):
                        return exp, sparql, query_count
        return '', '', query_count
    except:
        return '', '', query_count
    
def generate_variants(core):
    if not is_close(core):
        core = fix_core(core)
    bound_to_existed(core)

if __name__ == '__main__':
    core = '(AND (JOIN (R member_of_sports_team) (JOIN instance_of american_football_player) (JOIN instance_of sports_club))'
    generate_variants(core)
