#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import sys
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, ROOT_PATH)
from general_answer_model.src.solution import *
from general_answer_model.src.utils import *
import re


def split_formula(s):
    s = clean(s)
    result = []
    win = ''
    for i in s:
        if(i in ['+', '-'] and win != ''):
            result.append(win)
            win = i
        else:
            win += i
    result.append(win)
    return result

def extract_number(s):
    return re.findall('-?\d+(?:\.\d+)?', s)

def extract_char_chi(s):
    return re.findall('[\u4e00-\u9fff]+', s)

def extract_char_eng(s):
    return re.findall("[a-zA-Z]+", s)    

def getKeyIdxMultiple(doc, query):
    w = len(query)
    result = []
    for st in range(len(doc)-w+1):
        if(doc[st: st+w]==query):
            result.append(st)
    return result

# 针对应用题中，设未知数解方程，能够处理多个解的情况，
#能够处理在某些答案后写"舍",写了”舍“字的答案被忽略
def extract_key_from_latex(stuAns, keywords):
    lst = []
    for k in keywords:
        n = len(k)
        for line in stuAns.split('\n') :
            for st in range(len(line)-n):
                win = line[st: st+n]
                if(win == k):
                    prefix = max(0, st-1)     
                    if((not line[prefix].isdigit()) and (not line[prefix] in ['+', '-','(', ')'])): # 前缀不是数字或者符号,目的是匹配出'x=123'这种pattern
                        suffix_st = st+n
                        p = suffix_st
                        while(p<len(line)):
                            if(line[p].isdigit() or line[p]=='-' or '舍' in line[p:p+3]):#处理学生在某个答案后写”舍“的情况
                                p += 1
                            else:
                                break
                        lst.append(line[prefix+1: p])
    real_answer_lst = [x for x in lst if '舍' not in x]
    return real_answer_lst

# 分式拓展器，能拓展分式的多种写法
def to_more_format_fenshi(input_frac):
    count = 0
    for s in input_frac:
        if(s=='-'):
            count += 1
    numbers = [abs(int(x)) for x in extract_number(input_frac)]
    if(len(numbers)==3):
        x1, x2, x3 = numbers
        if(count%2==0):
            return ['{}'.format(x1)+'\\frac{'+'{}'.format(x2)+'}{'+'{}'.format(x3)+'}', '{}'.format(x1)+'{'+'{}'.format(x2)+'/'+'{}'.format(x3)+'}',  '{}/{}'.format(x1*x3+x2, x3), '\\frac{'+'{}'.format(x1*x3+x2)+'}'+'{'+'{}'.format(x3)+'}']
        else:
            return ['-{}'.format(x1)+'\\frac{'+'{}'.format(x2)+'}{'+'{}'.format(x3)+'}', '-{}'.format(x1)+'{'+'{}'.format(x2)+'/'+'{}'.format(x3)+'}',  '-{}/{}'.format(x1*x3+x2, x3), '-\\frac{'+'{}'.format(x1*x3+x2)+'}'+'{'+'{}'.format(x3)+'}']
    elif(len(numbers)==2):
        x1, x2 = numbers
        if(x1<x2):
            if(count%2==0):
                return ['{'+'{}'.format(x1)+'/'+'{}'.format(x2)+'}',  '{}/{}'.format(x1, x2), '\\frac{'+'{}'.format(x1)+'}'+'{'+'{}'.format(x2)+'}']
            else:
                return ['-{'+'{}'.format(x1)+'/'+'{}'.format(x2)+'}',  '-{}/{}'.format(x1, x2), '-\\frac{'+'{}'.format(x1)+'}'+'{'+'{}'.format(x2)+'}'] 
        else:
            a = int(x1/x2)
            b = int(x1%x2)
            c = x2
            if(count%2==0):
                return ['\\frac{'+'{}'.format(x1)+'}{'+'{}'.format(x2)+'}', '{}'.format(a)+'{'+'{}'.format(b)+'/'+'{}'.format(c)+'}',  '{}/{}'.format(a*c+b, c),  '{}'.format(a)+'\\frac{'+'{}'.format(b)+'}{'+'{}'.format(c)+'}']
            else:
                return ['-\\frac{'+'{}'.format(x1)+'}{'+'{}'.format(x2)+'}', '-{}'.format(a)+'{'+'{}'.format(b)+'/'+'{}'.format(c)+'}',  '-{}/{}'.format(a*c+b, c), '-{}'.format(a)+'\\frac{'+'{}'.format(b)+'}{'+'{}'.format(c)+'}']
    else:
        return [input_frac]

# 把标准答案进行分割，返回list
def split_answer(answer):
    separator = ['或', '，', ',', ';', '；', '、', '和', '，'] 
    result = []
    win = ''
    for s in answer:
        if(s in separator):
            result.append(win)
            win = ''
        else:
            win += s
    result.append(win)
    return result

# 答案分类器
def getAnswerType(ans):
    numbers = extract_number(ans)
    char_chi = extract_char_chi(ans)
    char_eng = extract_char_eng(ans)
    if(len(numbers)==1 and len(char_chi)+len(char_eng)==0):
        return '单个数字'
    elif('frac' in ans):
        return '分式'
    elif(len(numbers)>1 and len(char_chi)+len(char_eng)==0):
        return '多个数字'
    elif(len(numbers)>0 and len(char_chi)>0):
        return '文本和数字'
    elif('begin{cases}' in ans):
        return 'latex'
    elif(len(numbers)==0 and len(char_eng)==0 and len(char_chi)>0):
        return '纯文本'
    elif(len(numbers)>=0 and len(char_eng)>0 and len(char_chi)==0):
        if(',' in ans or '，' in ans):
            return '多个公式'
        else:
            return '单个公式'
    else:
        return '未知'

# 通用模型接口函数
def jieda_general_answer_solution(stuAns, answer_string):
    '''
    Args:
    stuAns: 学生回答，即OCR输出文本，type为字符串
    answer_string: 标准答案,type为字符串,多个答案以;分隔

    Returns:
    答案正误：0 或者 1, 1正确 0错误
    答案行: 空list代表没找到答案行，非空list包含n个答案行, n>=1
    '''
    # 空字符串返回
    if(stuAns=='' or answer_string==''):
        return 0, []
    if(isinstance(answer_string, str)):
        answer_list = split_answer(clean(answer_string))
    elif(isinstance(answer_string, list)):
        answer_list = answer_string
    flag = [0 for _ in range(len(answer_list))]
    # 定位答案行
    answer_line = locateAnswerLine(stuAns, keywords=['答'])

    for i, answer in enumerate(answer_list):
        ans_type = getAnswerType(answer)
        if(ans_type in ['纯文本']):
            stuAns = clean(stuAns)
            flag[i] = solution_C(stuAns, None, [answer], None)
        elif(ans_type in ['分式']):
            stuAns = clean(stuAns)
            ans_with_diff_format = to_more_format_fenshi(answer)
            flag[i] = solution_E(stuAns, None, ans_with_diff_format, None)
        elif(ans_type in ['单个数字', '多个数字']):
            numbers = extract_number(answer)
            flag[i] = solution_C_match_single_digit(stuAns, None, numbers, None)
        elif(ans_type in ['文本和数字']):
            numbers = extract_number(answer)
            stuAns = clean(stuAns)
            flag[i] = solution_C(stuAns, None, numbers, None)
        elif(ans_type in ['latex']):
            key_part = extract_key_from_latex(answer,  ['t=', 'x=', 'y=', 'z='])
            stuAns = clean(stuAns)
            flag[i] = solution_C(stuAns, None, key_part, None)
        elif(ans_type in ['单个公式']):
            ans_list = split_formula(answer)
            flag[i] = solution_match_formula_multiple(stuAns, None, ans_list, None)
        elif(ans_type in ['多个公式']):
            ans_list = [[x.split('=')[-1]] for x in answer.split(',')]
            flag[i] = solution_I(stuAns, None, ans_list, None)
        else:
            flag[i] = 1
    return int(sum(flag)==len(flag)), answer_line