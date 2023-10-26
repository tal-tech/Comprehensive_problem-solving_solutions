#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from itertools import combinations
from procedure_model.src.procedure_utils import *
# rule suanshi
'''
def extend_suanshi(procedure):
    new_list = []
    pro = clean(procedure)
    new_pro = pro
    equal_index = pro.find('=')
    equal_left = pro[:equal_index]
    equal_right = pro[equal_index:]
    plus_number_left = re.findall(r"(?:\+)\d+\.?\d*",equal_left)
    minus_number_left = re.findall(r"(?:\-)\d+\.?\d*",equal_left)
    plus_number_right = re.findall(r"(?:\+)\d+\.?\d*",equal_right)
    minus_number_right = re.findall(r"(?:\-)\d+\.?\d*",equal_right)
    new_list.append(pro)
    for num in plus_number_left:
        new_list.append(equal_left.replace(num,'') + equal_right + num.replace('+','-'))
    for num in minus_number_left:
        new_list.append(equal_left.replace(num,'') + equal_right + num.replace('-','+'))
    for num in plus_number_right:
        new_list.append(equal_left + num.replace('+','-') + equal_right.replace(num,''))
    for num in minus_number_right:
        new_list.append(equal_left + num.replace('-','+') + equal_right.replace(num,''))
    if len(plus_number_left) > 1:
        for fanwei in range(len(plus_number_left)):
            yuansubiao = combinations(plus_number_left,fanwei+1)
            for yuansu in yuansubiao:
                new_left = equal_left
                new_right = equal_right
                for i in yuansu:
                    new_left = new_left.replace(i,'')
                    new_right += i.replace('+','-')
                new_list.append(new_left + new_right)
    if len(minus_number_left) >1:
        for fanwei in range(len(minus_number_left)):
            yuansubiao = combinations(minus_number_left,fanwei+1)
            for yuansu in yuansubiao:
                new_left = equal_left
                new_right = equal_right
                for i in yuansu:
                    new_left = new_left.replace(i,'')
                    new_right += i.replace('-','+')
                new_list.append(new_left + new_right)
    if len(plus_number_right) >1:
        for fanwei in range(len(plus_number_right)):
            yuansubiao = combinations(plus_number_right, fanwei+1)
            for yuansu in yuansubiao:
                new_left = equal_left
                new_right = equal_right
                for i in yuansu:
                    new_right = new_right.replace(i,'')
                    new_left += i.replace('+','-')
                new_list.append(new_left+new_right)
    if len(minus_number_right)>1:
        for fanwei in range(len(minus_number_right)):
            yuansubiao = combinations(minus_number_right,fanwei+1)
            for yuansu in yuansubiao:
                new_left = equal_left
                new_right = equal_right
                for i in yuansu:
                    new_right = new_right.replace(i,'')
                    new_left += i.replace('-','+')
                new_list.append(new_left+new_right)
    check = []
    for i in new_list:
        l = i[:i.find('=')]
        r = i[i.find('=')+1:]
        if int(eval(l)) == int(eval(r)):
            check.append(i)
    return list(set(new_list))
'''
def extend_suanshi(procedure):
    new_list = []
    pro = clean(procedure)
    new_list.append(pro)
    temps = re.findall('\(\d+[\s\S]*\d+\)',pro)
    if len(temps)>0:
        for i in temps:
            new_list.append(pro.replace(i,str(eval(i))))
    return new_list
def calculate_similarity_suanshi(procedure_list, stu_answer_row):
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
'''
def rule_suanshi(stuAns_row, procedure, thres):
    procedure_list = extend_suanshi(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_suanshi(procedure_list, row) >= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide
'''

def rule_suanshi(stuAns_row, procedure, thres=0.7):
    procedure_list = [clean(procedure)]
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity(procedure_list, row) >= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide