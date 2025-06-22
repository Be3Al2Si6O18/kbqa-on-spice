import openai
import json
import os
from utils.process_file import process_file, process_file_node, process_file_rela, process_file_test
from time import sleep
import re
import logging
import argparse
import random
import itertools
from utils.parse_expr import expression_to_sparql, ParseError
from retriever.semantic_retriever import SemanticRetriever
from utils.execute_query import execute_query
import itertools
import numpy as np
import json
import asyncio
from utils.llm_call import LLM_Call

random.seed(1008)

exp_name = 'baseline_top1'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

entity_retriever = SemanticRetriever('entity')
relation_retriever = SemanticRetriever('relation')
rela_mid_to_faiss_index = {mid: i for i, mid in enumerate(relation_retriever.mid_list)}
rela_fn_to_faiss_index = {fn: i for i, fn in enumerate(relation_retriever.fn_list)}
wikidata_mid_to_fn = json.load(open('data/wikidata_mid_to_fn.json', 'r'))

type_retriever = SemanticRetriever('type')
type_mid_to_faiss_index = {mid: i for i, mid in enumerate(type_retriever.mid_list)}
type_fn_to_faiss_index = {mid: i for i, mid in enumerate(type_retriever.fn_list)}

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("time recoder")

API_KEYS = ['sk-54964e5c3b8c4998a74f7d3e35b618ac', 'sk-785362add9834d1da5907f4621dc90a8']
API_URL = 'https://api.deepseek.com'
API_MODEL = 'deepseek-chat'

api_pool = [(k, API_URL, API_MODEL) for k in API_KEYS]

LLM = LLM_Call(api_pool=api_pool, openai_params = {'temperature': 0, 'max_tokens': 512})

def select_shot_prompt_train(train_data_in, shot_number):
    random.shuffle(train_data_in)
    compare_list = ["LT", "LE", "EQ", "GE", "GT", "ARGMIN", "ARGMAX"]
    if shot_number == 1:
        selected_quest_compose = [train_data_in[0]["question"]]
        selected_quest_compare = [train_data_in[0]["question"]]
        selected_quest_mix = [train_data_in[0]["question"]]
    else:
        selected_quest_compose = []
        selected_quest_compare = []
        each_type_num = shot_number // 2
        for data in train_data_in:
            if any([x in data['s_expression'] for x in compare_list]):
                selected_quest_compare.append(data["question"])
                if len(selected_quest_compare) == each_type_num:
                    break
        for data in train_data_in:
            if not any([x in data['s_expression'] for x in compare_list]):
                selected_quest_compose.append(data["question"])
                if len(selected_quest_compose) == each_type_num:
                    break
        mix_type_num = each_type_num // 3
        selected_quest_mix = selected_quest_compose[:mix_type_num] + selected_quest_compare[:mix_type_num]
    logger.info("selected_quest_compose: {}".format(selected_quest_compose))
    logger.info("selected_quest_compare: {}".format(selected_quest_compare))
    logger.info("selected_quest: {}".format(selected_quest_mix))
    return selected_quest_compose, selected_quest_compare, selected_quest_mix

# 把string中的mid替换为friendly name, 调用时, 这里的string为prompt中的s_expression
def sub_mid_to_fn(string):
    seg_list = string.split()
    for i in range(len(seg_list)):
        token = seg_list[i].strip(')(')
        if token.startswith('P') or token.startswith('Q'):
            fn = wikidata_mid_to_fn.get(token, "unknown_entity")
            fn = fn.replace(' ', '_')
            seg_list[i] = seg_list[i].replace(token, fn)
    new_string = ' '.join(seg_list)
    return new_string

def type_generator(questions, prompt_type):
    sleep(1)
    messages_list = []
    for question in questions:
        messages = [{'role': 'system', 'content': "Your task is to predict the type of the given question. The multi-turn dialogue is separated by [SEP]. Only output the type of the last question. Just answer 'Composition' or 'Comparison'"}] \
            + prompt_type \
            + [{'role': 'user', 'content': question}]
        messages_list.append(messages)

    resp_list = asyncio.run(LLM._batch_generate_async(messages_list))
    gene_types = [resp.choices[0].message.content for resp in resp_list]
    return gene_types

def ep_generator(questions, types, prompt_compose, prompt_compare):
    sleep(1)
    messages_list = []
    for i, question in enumerate(questions):
        prompt = prompt_compare if types[i] == 'Comparison' else prompt_compose
        messages = [{'role': 'system', 'content': "Your task is to predict the logical form corresponding to the given question. The multi-turn dialogue is separated by [SEP]. Only output the logical form for the last question. Format the logical form exactly as shown in the examples. Respond directly with the logical form only — no explanations or clarifications."}] \
            + prompt \
            + [{'role': 'user', 'content': question}]
        messages_list.append(messages)

    resp_list = asyncio.run(LLM._batch_generate_async(messages_list))
    gene_exps = [resp.choices[0].message.content for resp in resp_list]
    return gene_exps


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

def calculate_type_similarity(vec1, type_mid):
    def get_vector_from_mid(type_mid):
        if type_mid not in type_retriever.mid_list:
            type_mid = type_retriever.semantic_search(wikidata_mid_to_fn[type_mid])[0][1]
        return type_retriever.index.reconstruct(type_mid_to_faiss_index[type_mid])

    vec2 = get_vector_from_mid(type_mid)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def bound_to_existed(s_expression, question_type):
    query_count = 0
    expression_segment = s_expression.split(" ")
    expression_segment_copy = expression_segment.copy()

    type_count = 0
    type_replace_dict = {}

    if question_type not in ['Comparative Reasoning (Count) (All)', 'Comparative Reasoning (All)', 'Quantitative Reasoning (All)', 'Logical Reasoning (All)']:
        for i, seg in enumerate(expression_segment):
            processed_seg = seg.strip(')')
            if i > 0 and expression_segment[i - 1] == 'instance_of':
                type_count += 1
                expression_segment_copy[i - 1] = 'P31'
                type_replace_dict[i] = f'?t{type_count}'
                expression_segment_copy[i] = type_replace_dict[i] + ')' * (len(seg) - len(processed_seg))
                if type_count >= 2:
                    break

    enti_replace_dict = {}
    for i, seg in enumerate(expression_segment):
        processed_seg = seg.strip(')')
        if processed_seg[0] != '(' and not processed_seg.isdigit() and not (i > 0 and expression_segment[i - 1] in ['(R', '(JOIN'] or i > 1 and expression_segment[i - 2] == '(IS_TRUE') and not i in type_replace_dict:
            enti_replace_dict[i] = [mid for score, mid, fn in entity_retriever.semantic_search(processed_seg)]
    if len(enti_replace_dict) > 4:
        top_k = 1
    elif len(enti_replace_dict) > 2:
        top_k = 1
    else:
        top_k = 1

    for i in enti_replace_dict:
        enti_replace_dict[i] = enti_replace_dict[i][:top_k]
    
    # print({id: [wikidata_mid_to_fn[mid] for mid in mids] for id, mids in enti_replace_dict.items()})
    
    combinations = list(enti_replace_dict.values())
    all_iters = list(itertools.product(*combinations)) # 所有可能的实体替换方案
    enti_index = list(enti_replace_dict.keys()) # 待替换实体在expression_segment中的index
    for iters in all_iters:
        for i in range(len(iters)):
            cur_enti = expression_segment[enti_index[i]]
            suffix = ')' * (len(cur_enti) - len(cur_enti.strip(')')))
            expression_segment_copy[enti_index[i]] = iters[i] + suffix

        rela_replace_dict = {}
        for j, seg in enumerate(expression_segment):
            processed_seg = seg.strip(')')
            if processed_seg[0] != '(' and not processed_seg.isdigit() and (j > 0 and expression_segment[j - 1] in ['(R', '(JOIN'] or j > 1 and expression_segment[j - 2] == '(IS_TRUE') and not (j + 1) in type_replace_dict:
                rela_replace_dict[j] = [mid for score, mid, fn in relation_retriever.semantic_search(processed_seg)]
        if len(rela_replace_dict) > 4:
            top_k = 1
        elif len(rela_replace_dict) > 2:
            top_k = 1
        else:
            top_k = 1

        for j in rela_replace_dict:
            rela_replace_dict[j] = rela_replace_dict[j][:top_k]
        
        # print({id: [wikidata_mid_to_fn[mid] for mid in mids] for id, mids in rela_replace_dict.items()})

        combinations_rela = list(rela_replace_dict.values())
        all_iters_rela = list(itertools.product(*combinations_rela))
        rela_index = list(rela_replace_dict.keys())
        for iter_rela in all_iters_rela:
            for k in range(len(iter_rela)):
                cur_rela = expression_segment[rela_index[k]]
                suffix = ')' * (len(cur_rela) - len(cur_rela.strip(')')))
                expression_segment_copy[rela_index[k]] = iter_rela[k] + suffix
            final = " ".join(expression_segment_copy)
            added = add_reverse(final) # 反转关系生成变体

            # 遍历added，首次能够查询到结果时，返回查询结果
            for exp in added:
                # print(exp)
                if type_replace_dict:
                    # if len(type_replace_dict) > 4:
                    #     return '', '', query_count
                    try:
                        sparql = expression_to_sparql(exp)
                        if sparql:
                            sparql = f'SELECT DISTINCT {' '.join(type_replace_dict.values())} ' + sparql[sparql.find('WHERE'):]
                        query_count += 1
                        possible_types = execute_query(sparql, multi_var=True)
                        # print([[wikidata_mid_to_fn[mid] for mid in binding] for binding in possible_types])
                        if isinstance(possible_types, list) and not possible_types or not possible_types[0]:
                            continue
                        
                        indexes = list(type_replace_dict.keys())
                        for type_id in range(len(type_replace_dict)):
                            index = indexes[type_id]
                            seg = expression_segment[index]
                            processed_seg = seg.strip(')')
                            possible_type = set([binding[type_id] for binding in possible_types])
                            seg_fn = type_retriever.semantic_search(processed_seg)[0][2] if processed_seg not in type_retriever.fn_list else processed_seg
                            seg_vector = type_retriever.index.reconstruct(type_fn_to_faiss_index[seg_fn])
                            type_similarity = [(typ, calculate_type_similarity(seg_vector, typ)) for typ in possible_type if typ in wikidata_mid_to_fn]
                            # print([(wikidata_mid_to_fn[mid], score) for mid, score in type_similarity])
                            most_similar_type, highest_similarity = max(type_similarity, key=lambda x: x[1])
                            possible_types = [binding for binding in possible_types if binding[type_id] == most_similar_type]
                            exp = exp.replace(f'?t{type_id + 1}', most_similar_type)
                    except:
                        return '', '', query_count
                sparql = expression_to_sparql(exp)
                query_count += 1
                answer = execute_query(sparql)
                if query_count >= 100:
                    return '', '', query_count
                if isinstance(answer, bool) or (answer and not (answer[0].isnumeric() and answer[0] == '0')):
                    return exp, sparql, query_count
    return '', '', query_count

def all_combiner_evaluation(all_data, prompt_compose, prompt_compare, prompt_type):

    os.makedirs(f'output/{exp_name}', exist_ok=True)
    output_file = f'output/{exp_name}/prediction.json'
    if os.path.exists(output_file):
        all_results = json.load(open(output_file, 'r'))
    else:
        all_results = []

    begin = len(all_results)

    batch_size = 20
    batches = [all_data[i: i + batch_size] for i in range(begin, len(all_data), batch_size)]
    
    for k, batch in enumerate(batches):
        questions = [data["question"] for data in batch]
        gene_types = type_generator(questions, prompt_type)
        gene_exps = ep_generator(questions, gene_types, prompt_compose, prompt_compare)

        for i in range(len(batch)):
            data = batch[i]
            logger.info("==========")
            logger.info("{}/{}".format(begin + k * batch_size + i + 1, len(all_data)))
            logger.info("data[id]: {}".format(data["turnID"]))
            logger.info("data[question]: {}".format(data["question"]))
            logger.info("data[exp]: {}".format(data["s_expression"]))
            s_expression_fn = sub_mid_to_fn(data["s_expression"])
            data['s_expression_fn'] = s_expression_fn
            logger.info("data[exp_fn]: {}".format(s_expression_fn))
            logger.info("gene_type: {}".format(gene_types[i]))
            logger.info("gene_exp: {}".format(gene_exps[i]))
            data['predicted_s_expression_fn'] = gene_exps[i]

            expr, sparql, sparql_attempt_count = bound_to_existed(gene_exps[i], data['question_type'])
            
            logger.info("S-expression: {}".format(expr))
            logger.info("SPARQL: {}".format(sparql))
            logger.info("SPARQL attempt count: {}".format(sparql_attempt_count))
            data['predicted_s_expression'] = expr
            data['sparql_attempt_count'] = sparql_attempt_count
            data['actions'] = sparql
            all_results.append(data)

        json.dump(all_results, open(f'output/{exp_name}/prediction.json', 'w'), indent=2)
    # json.dump(all_results, open(f'output/{exp_name}/prediction.json', 'w'), indent=2)

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--shot_num', type=int, metavar='N',
                        default=40, help='the number of shots used in in-context demo')
    parser.add_argument('--train_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_train.json", help='training data path')
    parser.add_argument('--eva_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_dev.json", help='evaluation data path')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dev_data = process_file(args.eva_data_path)
    train_data = process_file(args.train_data_path)

    que_to_s_dict_train = {data["question"]: data["s_expression"] for data in train_data}
    selected_quest_compose, selected_quest_compare, selected_quest_mix = select_shot_prompt_train(train_data, args.shot_num)

    prompt_compose = []
    for que in set(selected_quest_compose) | set(selected_quest_mix):
        if not que_to_s_dict_train[que]:
            continue
        prompt_compose.append({'role': 'user', 'content': que})
        prompt_compose.append({'role': 'assistant', 'content': sub_mid_to_fn(que_to_s_dict_train[que])})
    prompt_compare = []
    for que in set(selected_quest_compare) | set(selected_quest_mix):
        if not que_to_s_dict_train[que]:
            continue
        prompt_compare.append({'role': 'user', 'content': que})
        prompt_compare.append({'role': 'assistant', 'content': sub_mid_to_fn(que_to_s_dict_train[que])})

    prompt_type = []
    all_ques = selected_quest_compose + selected_quest_compare
    random.shuffle(all_ques)
    for que in all_ques:
        prompt_type.append({'role': 'user', 'content': que})
        prompt_type.append({'role': 'assistant', 'content': "Composition" if que in selected_quest_compose else "Comparison"})

    all_combiner_evaluation(dev_data, prompt_compose, prompt_compare, prompt_type)

if __name__=="__main__":
    main()