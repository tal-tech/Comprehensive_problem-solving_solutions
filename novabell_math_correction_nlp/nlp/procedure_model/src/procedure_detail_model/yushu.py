#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *

# 除法余数
def extend_yushu(procedure):
    new_list = []
    if ('\\div' in procedure) & ('余' in procedure):
        new_list.append(re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]|\\div', procedure) + re.findall(r'\d+(?:\.\d+)?',procedure))
        new_list.append(re.findall(r'[\u4E00-\u9FA5]',procedure) + re.findall(r'\d+',procedure))
    return [set(x) for x in new_list]
def calculate_similarity_yushu(procedure, stu_answer_row):
    pro = clean(procedure)
    list_procedure = extract_operator_and_number(pro)
    list_ans = extract_operator_and_number(stu_answer_row)
    intersection = len(set(list_procedure).intersection(list_ans))
    union = len(set(list_procedure)) + len(set(list_ans)) - intersection
    if union == 0:
        list_procedure = [x for x in pro]
        list_ans = [x for x in stu_answer_row]
        intersection = len(set(list_procedure).intersection(list_ans))
        union = len(set(list_procedure)) + len(set(list_ans)) - intersection
    jaccard = intersection/union
    return jaccard
def rule_yushu(stuAns_row, procedure, thres):
    procedure_fuzhu = extend_yushu(procedure)
    pro = clean(procedure)
    decide = 0
    row = clean(stuAns_row)
    if pro in row:
        decide = 1
    else:
        if calculate_similarity_yushu(pro,row)>=thres:
            decide = 1
        else:
            if (procedure_fuzhu!=[]) & (any(all(x in row for x in rule) for rule in procedure_fuzhu)):
                decide = 1
            else:
                plus_rule = [extract_operator_and_number(procedure)]
                if extract_operator_and_number(row) in plus_rule:
                    decide = 1
                else:
                    decide = 0
    return decide

