#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from functools import reduce
from itertools import permutations
from procedure_model.src.procedure_utils import *

# 公倍数
def lcm(a, b):
    if a > b:
        greater = a
    else:
        greater = b
    while True:
        if greater % a == 0 and greater % b == 0:
            lcm = greater
            break
        greater += 1
    return lcm
def get_lcm_for(your_list):
    return reduce(lambda x, y: lcm(x, y), your_list)
def perm(s):
    s1 = []
    for item in permutations(s):
        t = []
        for a in item:
            t.append(a)
        s1.append(t)
    return s1
def extend_gongbeishu(procedure):
    new_list = []
    pro = clean(procedure)
    new_pro = pro
    temps = re.findall(r'\[\d+[\s\S]*\d+\]',pro)
    for temp in temps:
        if len(re.findall(r'[^\x00-\xff]', temp)) != 0:
            continue
        res = eval(temp)
        nums = re.findall(r'\d+', temp)
        if len(res) == len(nums):  # 公倍数
            new_list.append(pro.replace(temp, ','.join(nums)+'的公倍数'))
            #new_list.append('='.join([temp,str(get_lcm_for(res))]))
            if temp != pro:
                new_list.append(pro.replace(temp, str(get_lcm_for(res))))
                new_pro = new_pro.replace(temp, str(get_lcm_for(res)))
        elif len(res) == 1 and temp != pro:
            new_list.append(pro.replace(temp, str(res[0])))
            new_pro = new_pro.replace(temp, str(res[0]))
        #if '=' not in pro:
            #new_list.append('='.join([pro,new_pro])) # 暂时comment，之后可以释放封印
        new_list.append(new_pro)
        #new_list += temps #暂时comment，之后可以释放封印
    return list(set([pro] + new_list))
def extend_gongbeishu_new(procedure):
    new_list = []
    pro = clean(procedure)
    new_pro = pro
    temps = re.findall(r'\[\d+[\s\S]*\d+\]',pro)
    for temp in temps:
        if len(re.findall(r'[^\x00-\xff]', temp)) != 0:
            continue
        res = eval(temp)
        nums = re.findall(r'\d+', temp)
        if len(res) == len(nums):  # 公倍数
            extended_nums = perm(nums)
            for arr in extended_nums:
                new_list.append(pro.replace(temp, ','.join(arr)+'的公倍数'))
                new_list.append(pro.replace(temp, '['+','.join(arr)+']'))
            #new_list.append('='.join([temp,str(get_lcm_for(res))]))
            if temp != pro:
                new_list.append(pro.replace(temp, str(get_lcm_for(res))))
                new_pro = new_pro.replace(temp, str(get_lcm_for(res)))
        elif len(res) == 1 and temp != pro:
            new_list.append(pro.replace(temp, str(res[0])))
            new_pro = new_pro.replace(temp, str(res[0]))
        #if '=' not in pro:
            #new_list.append('='.join([pro,new_pro])) # 暂时comment，之后可以释放封印
        new_list.append(new_pro)
        #new_list += temps #暂时comment，之后可以释放封印
    return list(set([pro] + new_list))
def extract_operator_and_number_gongbeishu(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s) + re.findall(r'\d+(?:\\d+)?', s)
def calculate_similarity_gongbeishu(procedure_list, stu_answer_row):
    '''
    procedure_list:
        type: list
        sample: ['1+1=2','2-1=1']
    stu_answer_row:
        type: str
        sample: '1+1=2'
    '''
    tmp = []
    for procedure in procedure_list:
        list_procedure = extract_operator_and_number_gongbeishu(procedure)
        list_ans = extract_operator_and_number_gongbeishu(stu_answer_row)
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
def rule_gongbeishu(stuAns_row, procedure, thres):
    # procedure_list = extend_gongbeishu(procedure)
    procedure_list = extend_gongbeishu_new(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_gongbeishu(procedure_list,row)>= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide

