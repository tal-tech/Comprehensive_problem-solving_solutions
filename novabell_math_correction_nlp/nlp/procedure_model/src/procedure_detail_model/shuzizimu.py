#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *
from itertools import permutations


def perm(s):
    s1 = []
    for item in permutations(s):
        t = []
        for a in item:
            t.append(a)
        s1.append(t)
    return s1

# def exchange_order(procedure):
#     new_list = []
#     pro = procedure
#     match = re.findall(r'(?:(?:\d*[a-zA-Z]+|\d+[a-zA-Z]*)[\+-])+(?:\d*[a-zA-Z]+|\d+[a-zA-Z]*)', pro)
#     for m in match:
#         nums = re.findall(r'[\+-](?:\d*[a-zA-Z]+|\d+[a-zA-Z]*)', m)
#         nums = [x if len(re.findall(r'[\+-]', x))!=0 else '+'+x for x in nums]
#         extended_nums = perm(nums)
#         for arr in extended_nums:
#             new_list.append(pro.replace(m, ''.join(arr)))

# 10(10a+5b+5c)
def extend_shuzizimu1(procedure):
    pro = procedure
    new_list = []
    new_pro1 = pro
    new_pro2 = pro
    match = re.findall(r'[1-9][0-9]*\((?:\d*[a-zA-Z]+[\+\-])+\d*[a-zA-Z]+\)', pro)
    for i in range(len(match)):
        num = re.findall(r'([1-9][0-9]*)\([^\)]*\)',match[i])[0]
        other = re.findall(r'[1-9][0-9]*(\([^\)]*\))', match[i])[0]
        temp1 = pro.replace(match[i], other+'*'+num)
        temp2 = pro.replace(match[i], num+'*'+other)
        new_pro1 = new_pro1.replace(match[i], other+'*'+num)
        new_pro2 = new_pro2.replace(match[i], num+'*'+other)
        new_list.append(temp1)
        new_list.append(temp2)
        for j in range(i+1, len(match)):
            num1 = re.findall(r'([1-9][0-9]*)\([^\)]*\)',match[j])[0]
            other1 = re.findall(r'[1-9][0-9]*(\([^\)]*\))', match[j])[0]
            temp1 = temp1.replace(match[j], other1+'*'+num1)
            temp2 = temp2.replace(match[j], num1+'*'+other1)
            new_list.append(temp1)
            new_list.append(temp2)
    new_list.append(new_pro1)
    new_list.append(new_pro2)
    return list(set([pro]+new_list))

# 10*(10a+5b+5c) or 10/(10a+5b+5c) or (10a+5b+5c)*10 or (10a+5b+5c)/5
# 变形扩展
def extend_shuzizimu2(procedure):
    new_list = []
    pro = procedure
    match1 = re.findall(r'\d+\s*(?:\*|\/)\s*\((?:\d*[a-zA-Z]+[\+-])+\d*[a-zA-Z]+\)', pro) 
    for m in match1:
        left = re.split(r'\s*(\*|\/)\s*', m)[0]
        mid = re.split(r'\s*(\*|\/)\s*', m)[1]
        right = re.split(r'\s*(\*|\/)\s*', m)[2]
        letters = re.findall(r'\d*[a-zA-Z]+', right)
        fuhao = re.findall(r'[\+-]', right)
        new_str1 = new_str2 = new_str3 = ''
        for i in range(len(fuhao)):
            new_str1 += left+mid+letters[i]+fuhao[i]
            num = re.findall(r'(\d+)[a-zA-Z]+', letters[i])
            letter = re.findall(r'\d+([a-zA-Z]+)', letters[i])
            if len(num) > 0:
                new_str2 += str(eval(left+mid+num[0]))+'*'+letter[0]+fuhao[i]
                new_str3 += letter[0]+fuhao[i] if eval(left+mid+num[0])==1 else str(eval(left+mid+num[0]))+letter[0]+fuhao[i]
            else:
                new_str2 += left+mid+letters[i]+fuhao[i]
                new_str3 += left+letters[i]+fuhao[i] if mid=='*' else left+mid+letters[i]+fuhao[i]
        new_str1 += left+mid+letters[-1]
        num = re.findall(r'(\d+)[a-zA-Z]+', letters[-1])
        letter = re.findall(r'\d+([a-zA-Z]+)', letters[-1])
        if len(num) > 0:
            new_str2 += str(eval(left+mid+num[0]))+'*'+letter[0]
            new_str3 += letter[0] if eval(left+mid+num[0])==1 else str(eval(left+mid+num[0]))+letter[0]
        else:
            new_str2 += left+mid+letters[-1]
            new_str3 += left+letters[-1] if mid=='*' else left+mid+letters[-1]
        if m == pro or pro[pro.index(m)-1]=='=' or (len(pro)>pro.index(m)+len(m) and pro[pro.index(m)+len(m)])=='=':
            new_list.append(pro.replace(m, new_str1))
            new_list.append(pro.replace(m, new_str2))
            new_list.append(pro.replace(m, new_str3))
        else:
            new_list.append(pro.replace(m, '('+new_str1+')'))
            new_list.append(pro.replace(m, '('+new_str2+')'))
            new_list.append(pro.replace(m, '('+new_str3+')'))
        new_list.append(pro.replace(m, right+mid+left))
        
    # (10a+5b+5c)*10 or (10a+5b+5c)/5
    match2 = re.findall(r'\((?:\d*[a-zA-Z]+[\+-])+\d*[a-zA-Z]+\)\s*(?:\*|\/)\s*\d+', pro)
    for m in match2:
        left = re.split(r'\s*(\*|\/)\s*', m)[0]
        mid = re.split(r'\s*(\*|\/)\s*', m)[1]
        right = re.split(r'\s*(\*|\/)\s*', m)[2]
        letters = re.findall(r'\d*[a-zA-Z]+', left)
        fuhao = re.findall(r'[\+-]', left)
        new_str1 = new_str2 = new_str3 = ''
        for i in range(len(fuhao)):
            new_str1 += letters[i]+mid+right+fuhao[i]
            num = re.findall(r'(\d+)[a-zA-Z]+', letters[i])
            letter = re.findall(r'\d+([a-zA-Z]+)', letters[i])
            if len(num) > 0:
                new_str2 += str(eval(num[0]+mid+right))+'*'+letter[0]+fuhao[i]
                new_str3 += letter[0]+fuhao[i] if eval(num[0]+mid+right)==1 else str(eval(num[0]+mid+right))+letter[0]+fuhao[i]
            else:
                new_str2 += letters[i]+mid+right+fuhao[i]
                new_str3 += right+letters[i]+fuhao[i] if mid=='*' else letters[i]+mid+right+fuhao[i]
        new_str1 += letters[-1]+mid+right
        num = re.findall(r'(\d+)[a-zA-Z]+', letters[-1])
        letter = re.findall(r'\d+([a-zA-Z]+)', letters[-1])
        if len(num) > 0:
            new_str2 += str(eval(num[0]+mid+right))+'*'+letter[-1]
            new_str3 += letter[-1] if eval(num[0]+mid+right)==1 else str(eval(num[0]+mid+right))+letter[-1]
        else:
            new_str2 += letters[-1]+mid+right
            new_str3 += right+letters[-1] if mid=='*' else letters[-1]+mid+right
        if m == pro or pro[pro.index(m)-1]=='=' or (len(pro)>pro.index(m)+len(m) and pro[pro.index(m)+len(m)])=='=':
            new_list.append(pro.replace(m, new_str1))
            new_list.append(pro.replace(m, new_str2))
            new_list.append(pro.replace(m, new_str3))
        else:
            new_list.append(pro.replace(m, '('+new_str1+')'))
            new_list.append(pro.replace(m, '('+new_str2+')'))
            new_list.append(pro.replace(m, '('+new_str3+')'))
        new_list.append(pro.replace(m, right+mid+left))
    return list(set([pro]+new_list))

def calculate_similarity_shuzizimu(procedure_list, stu_answer_row):
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
        list_procedure = extract_operator_and_shuzizimu(procedure)
        list_ans = extract_operator_and_shuzizimu(stu_answer_row)
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

def rule_shuzi_zimu(stuAns_row, procedure, thres):
    procedure = clean(procedure)
    procedure_list = []
    procedure_list += extend_shuzizimu1(procedure)
    procedure_list += extend_shuzizimu2(procedure)
    procedure_list = list(set(procedure_list))
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_shuzizimu(procedure_list, row) >= thres:
            decide = 1
        else:
        #     plus_rule = [extract_operator_and_number(x) for x in procedure_list]
        #     if extract_operator_and_number(row) in plus_rule:
        #         decide = 1
            plus_rule = [extract_operator_and_shuzizimu(x) for x in procedure_list]
            if extract_operator_and_shuzizimu(row) in plus_rule:
                decide = 1
    return decide

