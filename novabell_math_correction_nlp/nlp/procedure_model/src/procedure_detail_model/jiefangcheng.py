#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *

# rule jiefangcheng
def extend_jiefangcheng(procedure):
    new_list = []
    pro = clean(procedure)
    new_list.append(pro)
    if 'x' in pro:
        new_list.append(pro.replace('x','a'))
    if 'y' in pro:
        new_list.append(pro.replace('y','b'))
    if ('x' in pro) & ('y' in pro):
        propro = pro.replace('x','a')
        propro = propro.replace('y','b')
        new_list.append(propro)
    return new_list
def calculate_similarity_jiefangcheng(procedure_list, stu_answer_row):
    tmp = []
    for procedure in procedure_list:
        list_procedure = extract_operator_and_number(procedure)
        list_ans = extract_operator_and_number(stu_answer_row)
        intersection = len(set(list_procedure).intersection(list_ans))
        union = len(set(list_procedure)) + len(set(list_ans)) - intersection
        if union == 0:
            list_procedure = [x for x in procedure]
            list_ans = [x for x in stu_answer_row]
            intersection = len(set(list_procedure).intersection(list_ans))
            union = len(set(list_procedure)) + len(set(list_ans)) - intersection
        jaccard = intersection/union
        tmp.append(jaccard)
    return max(tmp)
def rule_jiefangcheng(stuAns_row, procedure, thres):
    procedure_list = extend_jiefangcheng(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_jiefangcheng(procedure_list, row) >= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide
