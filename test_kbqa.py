import openai
import json
from utils.process_file import process_file
from utils.parse_expr import ExprParser
from utils.execute_query import execute_query
from utils.utils import *
from time import sleep
import re
import os
import logging
from collections import Counter
import argparse
import random
from retriever.semantic_retriever import semantic_search

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("time recoder")

# def select_shot_prompt_train(train_data_in, shot_number):
#     selected_quest = random.sample(train_data_in, shot_number)
#     logger.info("selected_quest: {}".format(selected_quest))
#     return [quest['question'] for quest in selected_quest]

def select_shot_prompt_train(train_data_in, shot_number):
    random.shuffle(train_data_in)
    compare_list = ['LT', 'LE', 'EQ', 'GE', 'GT']

    selected_quest = {'verify': [], 'optimize': [], 'compare': [], 'count': [], 'simple': []}
    each_type_num = max(2 * shot_number // 5, 1)
    for data in train_data_in:
        if data['s_expression'].startswith('(ASK'): # verify
            quest_type = 'verify'
        elif data['s_expression'].startswith('(ARG'): # optimize
            quest_type = 'optimize'
        elif any([x in data['s_expression'] for x in compare_list]): # compare
            quest_type = 'compare'
        elif data['s_expression'].startswith('(COUNT'): # count
            quest_type = 'count'
        else: # simple
            quest_type = 'simple'
        if len(selected_quest[quest_type]) < each_type_num:
            selected_quest[quest_type].append(data['question'])
        if all([len(selected_quest[quest_type]) == each_type_num for quest_type in selected_quest]):
            break
    # return [question for quest_type in selected_quest for question in selected_quest[quest_type]]
    
    mix_type_num = each_type_num // 4
    selected_quest['mix'] = [question for quest_type in ['optimize', 'compare', 'count', 'simple'] for question in selected_quest[quest_type][:mix_type_num]]

    logger.info("selected_quest: {}".format(selected_quest))
    return selected_quest

# 把string中的mid替换为friendly name, 调用时, 这里的string为prompt中的s_expression
def sub_mid_to_fn(expression, wikidata_mid_to_fn_dict):
    seg_list = expression.split()
    for i in range(len(seg_list)):
        token = seg_list[i].strip(')(')
        if token.startswith('P') or token.startswith('Q'):
            fn = wikidata_mid_to_fn_dict.get(token, "unknown_entity")
            seg_list[i] = seg_list[i].replace(token, fn)
    new_expression = ' '.join(seg_list)
    return new_expression

def sub_fn_to_mid(expression):
    func_list = ['R', 'JOIN', 'AND', 'OR', 'DIFF', 'VALUES', 'DISTINCT', 'COUNT', 'GROUP_COUNT', 'GROUP_SUM', 'LT', 'LE', 'EQ', 'GE', 'GT', 'ARGMIN', 'ARGMAX', 'ALL', 'IS_TRUE']
    seg_list = expression.split()
    for i in range(len(seg_list)):
        token = seg_list[i].strip(')(')
        if token not in func_list and not token.isdigit():
            dis, mid, fn = semantic_search(token)[0]
            logger.info("linking: {} {}".format(fn, mid))
            seg_list[i] = seg_list[i].replace(token, mid)
    new_expression = ' '.join(seg_list)
    return new_expression

def type_generator(question, prompt_type, api_key, LLM_engine):
    sleep(1)
    got_result = False
    while got_result != True:
        try:
            openai.api_key = api_key
            resp = openai.chat.completions.create(
                model=LLM_engine,
                messages=[
                    {
                        'role': 'assistant',
                        'content': prompt_type
                    },
                    {
                        'role': 'user',
                        'content': " Question: " + question + "Type of the question: "
                    },
                    {
                        'role': 'user',
                        'content': "Just answer 'verify', 'optimize', 'compare', 'count' or 'simple', don't provide explanation or clarification"
                    }
                ],
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            got_result = True
        except:
            sleep(3)
    gene_type = resp.choices[0].message.content
    if len(gene_type) > 10:
        gene_type = 'simple'
    return gene_type


def ep_generator(question, selected_examples, temp, que_to_s_dict_train, wikidata_mid_to_fn_dict, api_key, LLM_engine):
    messages = [{'role': 'system', 'content': "Convert the given question into a precise S-expression logical form, strictly following the syntax and structure of the examples provided. Do not include any explanations or additional text in your response."}]
#     prompt = r"""
# Next, please transform the problems into logical forms. The logical forms here are composed of the following functions:
# (JOIN predicate object), which returns a set consisting of all subjects for which the (subject, predicate, object) triple is true. The sets here and below may contain duplicate elements.
# (JOIN (R predicate) subject), which returns a set consisting of all objects for which (the subject, predicate, object) triple is true. (R predicate) here represents the reversal of the relationship.
# (AND set1 set2 ...), which returns the intersection of several sets.
# (OR set1 set2 ...), which returns the union of several sets.
# (DIFF set1 set2), which returns the difference set of the two sets, that is, set1 - set2.
# (VALUES value1 value2 ...), which returns a set composed of values such as value1 and value2.
# (DISTINCT set), which returns the set after removing duplicates.
# (COUNT set), which returns the number of elements in the set.
# (GROUP_COUNT set), performs grouped counting and returns a dictionary in Python, where the keys and values are the elements in the set and their counts in the set respectively.
# (GROUP_SUM group_count1 group_count2), combines the results of grouped counting of two sets and returns a dictionary, where the keys and values are the elements and their total counts in the two sets respectively.
# # (LT dictionary value0), which selects the keys from the dictionary whose corresponding values are less than value0 and returns them as a new set. Similarly, LE, EQ, GE, GT represent less than or equal to, equal to, greater than or equal to, and greater than respectively.
# (ARGMIN dictionary), returns the key corresponding to the minimum value in the dictionary. Similarly, ARGMAX returns the key corresponding to the maximum value.
# (ASK (subject1 predicate1 object1) (subject2 predicate2 object2) ...), determines whether several triples are simultaneously true.
# Here are some examples:
# """
    for que in selected_examples:
        if not que_to_s_dict_train[que]:
            continue
        messages.append({'role': 'user', 'content': que})
        messages.append({'role': 'assistant', 'content': sub_mid_to_fn(que_to_s_dict_train[que], wikidata_mid_to_fn_dict)})
    messages.append({'role': 'user', 'content': question})

    got_result = False
    while got_result != True:
        try:
            openai.api_key = api_key
            resp = openai.chat.completions.create(
                model=LLM_engine,
                messages=messages,
                temperature=temp,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            got_result = True
        except:
            sleep(3)
    gene_exp = [exp.message.content for exp in resp.choices]
    return gene_exp

def struct_of_exp(exp):
    ans = []
    for token in exp.split():
        if token[0] == '(':
            ans.append(token)
        elif token[-1] == ')':
            n = token.count(')')
            ans.append('x')
            ans.append(token[-n:])
        else:    
            ans.append('x')
    return ans

def all_combiner_evaluation(data_batch, selected_quest, prompt_type,
                            temp, que_to_s_dict_train, wikidata_mid_to_fn_dict,
                            expr_parser,
                            api_key, LLM_engine, exp_name):
    correct_exp = 0
    total_exp = 0
    correct_exp_set = set()
    all_results = []
    for qa in data_batch:
        logger.info("==========")
        logger.info("data[id]: {}".format(qa["turnID"]))
        logger.info("data[question]: {}".format(qa["question"]))
        logger.info("data[exp]: {}".format(sub_mid_to_fn(qa["s_expression"], wikidata_mid_to_fn_dict)))

        gene_type = type_generator(qa["question"], prompt_type, api_key, LLM_engine)
        logger.info("gene_type: {}".format(gene_type))

        if gene_type == 'verify':
            selected_examples = selected_quest['verify']
        else:
            selected_examples = list(set(selected_quest[gene_type]) | set(selected_quest['mix']))

        gene_exps = ep_generator(qa["question"], selected_examples,
                                    temp, que_to_s_dict_train, wikidata_mid_to_fn_dict,
                                    api_key, LLM_engine)     

        for idx, gene_exp in enumerate(gene_exps):
            total_exp += 1
            # logger.info("================================================================")
            logger.info("gene_exp: {}".format(gene_exp))
            
            if struct_of_exp(gene_exp) == struct_of_exp(qa["s_expression"]):
                logger.info("correct structure")
                correct_exp += 1
                correct_exp_set.add(gene_exp)
            else:
                logger.info("wrong structure")
            
            expr = sub_fn_to_mid(gene_exp)
            logger.info("S-expression: {}".format(expr))
            try:
                sparql = expr_parser.parse_expr(expr)
                logger.info("SPARQL: {}".format(sparql))
            except:
                sparql = ''
                logger.info("Fail to parse: {}".format(expr))
            # try:
            #     results = execute_query(sparql)
            #     logger.info("Query results: {}".format(results))
            # except:
            #     logger.info("Fail to query: {}".format(expr))
            #     continue
            qa['actions'] = clean_prediction(sparql)
            all_results.append(qa)
    
    os.makedirs(f'output/{exp_name}', exist_ok=True)
    dump_json(all_results, f'output/{exp_name}/prediction.json')

    print(f"correct_exp: {correct_exp}")
    print(f"total_exp: {total_exp}")

def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--shot_num', type=int, metavar='N',
                        default=40, help='the number of shots used in in-context demo')
    parser.add_argument('--temperature', type=float, metavar='N',
                        default=0.3, help='the temperature of LLM')
    parser.add_argument('--api_key', type=str, metavar='N',
                        default=None, help='the api key to access LLM')
    parser.add_argument('--engine', type=str, metavar='N',
                        default="code-davinci-002", help='engine name of LLM')
    parser.add_argument('--retrieval', action='store_true', help='whether to use retrieval-augmented KB-BINDER')
    parser.add_argument('--train_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_train.json", help='training data path')
    parser.add_argument('--eva_data_path', type=str, metavar='N',
                        default="data/GrailQA/grailqa_v1.0_dev.json", help='evaluation data path')
    parser.add_argument('--exp_name', type=str, metavar='N',
                        default="test", help='experiment name')
    
    args = parser.parse_args()
    return args

def main():
    openai.base_url = 'http://222.29.156.145:8000/v1/'
    # openai.base_url = 'https://api.deepseek.com'
    args = parse_args()
    dev_data = process_file(args.eva_data_path)
    train_data = process_file(args.train_data_path)
    que_to_s_dict_train = {data["question"]: data["s_expression"] for data in train_data}
    
    logger.info("selecting prompt...")
    selected_quest = select_shot_prompt_train(train_data, args.shot_num)

    prompt_type = ''
    for quest_type in selected_quest:
        if quest_type != 'mix':
            for que in selected_quest[quest_type]:
                prompt_type = prompt_type + "Question: " + que + "\nType of the question: " + quest_type + "\n"

    wikidata_mid_to_fn_dict = load_bin('data/wikidata_mid_to_fn_dict.pickle')

    expr_parser = ExprParser()

    all_combiner_evaluation(dev_data, selected_quest, prompt_type,
                            args.temperature, que_to_s_dict_train, wikidata_mid_to_fn_dict,
                            expr_parser,
                            args.api_key, args.engine, args.exp_name)

if __name__=="__main__":
    main()