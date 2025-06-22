"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import pickle
import json
import os
import shutil

def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)

def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)

def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

def clean_prediction(prediction: str) -> str:
    prediction = f"{prediction}"
    prediction = prediction.replace("\n", " ")

    # remove "SPARQL query:" that is added due to few-shot examples
    prediction = prediction.replace("SPARQL query:", " ")

    # remove everything after separator "<\s>" (LLaMA few-shot)
    prediction = prediction.split("</s>")[0]

    # add spaces arount "."
    prediction = prediction.replace(".", " . ")

    # add a space before ?
    prediction = prediction.replace("?", " ?")

    # remove duplicate spaces
    prediction = " ".join(prediction.split())

    # remove leading and trailing spaces
    prediction = prediction.strip()

    return prediction

def close_parentheses(expression):
    left, right = 0, 0
    redundant_right_parentheses = []
    for index, char in enumerate(expression):
        if char == '(':
            left += 1
        elif char == ')':
            if right == left:
                redundant_right_parentheses.append(index)
            else:
                right += 1
    expression = ''.join([expression[i] for i in range(len(expression)) if i not in redundant_right_parentheses])
    if left < right:
        expression = expression + ')' * (right - left)
    return expression