#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from fuzzywuzzy import fuzz
from general_answer_model.src.utils import *

# 针对标答包含多个数字的情况，而且数字是简单的正整数，不可为分式、小数等
# 不适用于答案为一个数字的，因为会检查数字个数
# 先通过关键词定位答案行，然后从答案行提取数字列表，验证数字列表是否和标答列表严格相等
# 如果提取不到答案行，则直接验证学生文本里是否包含答案列表里的全部数字
# 提取数字的时候无法区分正负数，例如-10会被提取成10
def solution_A(stuAns, keywords, ans_list, wrong_lst):
    n = len(ans_list)
    answer_line, other_line = locateAnswerLine_new(stuAns, keywords, n) 
    if(len(answer_line)>0):
        for line in answer_line:
            numbers = extractNumbers(line)
            if(all(elemt in numbers for elemt in ans_list) and len(numbers)==n):
                return 1
        return 0
    else:
        numbers = extractNumbers(stuAns)
        if(all(elemt in numbers for elemt in ans_list)):
            return 1
    return 0


# 针对标答包含多个数字的情况，而且数字可以是整数、小数、正负号
# 不适用于答案为一个数字的，因为会检查数字个数
# 匹配是加上高频数字补丁
# 先通过关键词定位答案行，然后从答案行提取数字列表，验证数字列表是否和标答列表严格相等
# 如果提取不到答案行，则直接验证学生文本里是否包含答案列表里的全部数字
# 提取数字的时候无法区分正负数，例如-10会被提取成10
def solution_A_match_single_digit(stuAns, keywords, ans_list, wrong_lst):
    n = len(ans_list)
    answer_line, other_line = locateAnswerLine_new(stuAns, keywords, n) 
    if(len(answer_line)>0):
        for line in answer_line:
            numbers = extract_number(line)
            if(match_multiple_single_digit(line, ans_list) and len(numbers)==n):
                return 1
        return 0
    else:
        return match_multiple_single_digit(stuAns, ans_list)


# 针对计算题的解法，标答唯一只有一个数字的情况
#思路：提取最后一行作为答案行，如果答案行有等于号，等号后面是学生答案
#检查标答列表里是否被包含在学生答案里，并且学生答案不包含错误答案列表里任意元素
def solution_B(stuAns, keywords, ans_list, wrong_lst):
    answer_line = stuAns.split('\n')[-1] # 只取最后一行
    if(answer_line == ' '): # 如果最后一行后面有'\n'，则answer_line为' '，此时要获取倒数第二行
        answer_line = stuAns.split('\n')[-2]
    #answer_line = stuAns.split('\n')[-2:]
    answer_line = answer_line.replace(' ', '')
    if('=' in answer_line):
        ans = answer_line.split('=')[-1]
    else:
        ans = answer_line
    for s in ans_list:
        if(s in ans):
            if(not wrong_lst==None):
                for wrong in wrong_lst:
                    if(wrong in ans):
                        return 0
            return 1
    return 0


# 针对计算题的解法，标答唯一只有一个数字的情况
#思路：提取最后一行作为答案行，如果答案行有等于号，等号后面是学生答案
#检查标答列表里是否被包含在学生答案里，并且学生答案不包含错误答案列表里任意元素
def solution_B_extract_number(stuAns, keywords, ans_list, wrong_lst):
    answer_line = stuAns.split('\n')[-1] # 只取最后一行
    if(answer_line == ' '): # 如果最后一行后面有'\n'，则answer_line为' '，此时要获取倒数第二行
        answer_line = stuAns.split('\n')[-2]
    #answer_line = stuAns.split('\n')[-2:]
    answer_line = answer_line.replace(' ', '')
    if('=' in answer_line):
        ans = extract_number(answer_line.split('=')[-1])
    else:
        ans = extract_number(answer_line)
    for s in ans_list:
        if(s in ans):
            if(not wrong_lst==None):
                for wrong in wrong_lst:
                    if(wrong in ans):
                        return 0
            return 1
    return 0



# 硬匹配,多个答案需要全部都在，判断为正确
def solution_C(stuAns, keywords, ans_list, wrong_lst):
    #stuAns = ''.join([x for x in stuAns.split('\n') if ('x=' in x or 'y=' in x)])
    for ans in ans_list:
        if(not ans in stuAns):
            return 0
    return 1


# 硬匹配,多个答案需要全部都在，判断为正确
def solution_C_extract_number(stuAns, keywords, ans_list, wrong_lst):
    #stuAns = ''.join([x for x in stuAns.split('\n') if ('x=' in x or 'y=' in x)])
    stuAns_number = extract_number(stuAns)
    for ans in ans_list:
        if(not ans in stuAns_number):
            return 0
    return 1

# 硬匹配,多个答案需要全部都在，判断为正确
# 使用高频词补丁匹配
def solution_C_match_single_digit(stuAns, keywords, ans_list, wrong_lst):
    return match_multiple_single_digit(stuAns, ans_list)

# 基于solution_A的基础上增加了多个答案的顺序检测
# 适用于问题中包含多个主体的情况，例如大桶小桶分别多少个，答：大桶20个，小桶30个
# 先通过关键字定位答案行，抽取答案行数字，比较数字和数字个数是否和标答相等
# 在满足以上条件下会最后再检查顺序，例如 答：大桶30个，小桶20个 由于顺序写反了会被检查出来错误
def solution_D(stuAns, keywords, ans_dict, wrong_lst):
    ans_list = list(ans_dict.values())
    n = len(ans_list)
    key_idx = {k:0 for k in ans_dict.keys()}
    value_idx = {k:0 for k in ans_dict.values()}
    answer_line, _ = locateAnswerLine_new(stuAns, keywords, n) 
    if(len(answer_line)>0):
        for line in answer_line:
            numbers = extractNumbers(line)
            if(all(elemt in numbers for elemt in ans_list) and len(numbers)==n):
                for key in ans_dict:
                    if(not key in line):
                        return 1
                # 检查顺序
                for key in ans_dict:
                    this_key_idx = getKeyIdx(line, key)
                    key_idx[key] = this_key_idx
                    this_value_idx = getKeyIdx(line, ans_dict[key])
                    value_idx[ans_dict[key]] = this_value_idx

                tmp1 = [x[0] for x in sorted(key_idx.items(), key=lambda x: x[1])]
                tmp2 = [x[0] for x in sorted(value_idx.items(), key=lambda x: x[1])]
                result = {}
                for i, j in zip(tmp1, tmp2):
                    result[i] = j
                return int(result == ans_dict)
            else:
                return 0
    else:
        numbers = extractNumbers(stuAns)
        if(all(elemt in numbers for elemt in ans_list)):
            return 1
    return 0


# 硬匹配，学生答案包含答案列表里任意就判为正确
def solution_E(stuAns, keywords, ans_list, wrong_lst):
    for ans in ans_list:
        if(ans in stuAns):
            return 1
    return 0


# 针对标答为几月几日的情况
# 先通过关键词定位答案行，然后检查答案行是否包含标答
# 如果提取不到答案行，则直接验证学生文本里是否包含标答
def solution_F(stuAns, keywords, ans_list, wrong_lst):
    n = len(ans_list)
    answer_line, _ = locateAnswerLine_new(stuAns, keywords, 3) 
    if(len(answer_line)>0):
        for line in answer_line:
            line = line.replace('\n', '').replace(' ', '')
            for ans in ans_list:
                if(ans in line):
                    return 1
        return 0
    else:
        for ans in ans_list:
            if(ans in stuAns):
                return 1
    return 0

# 针对标答包含多个数字的情况，而且数字是简单的正整数，不可为分式、小数等
# 先通过关键词定位答案行，然后从答案行提取数字列表，验证数字列表是否和标答列表严格相等
# 如果提取不到答案行，则直接验证学生文本里是否包含答案列表里的全部数字
# 提取数字的时候无法区分正负数，例如-10会被提取成10
def solution_G(stuAns, keywords, ans_list, wrong_lst):
    n = len(ans_list)
    answer_line, _ = locateAnswerLine_new(stuAns, keywords, n) 
    if(len(answer_line)>0):
        for line in answer_line:
            numbers = extractNumbers(line)
            if(all(elemt in numbers for elemt in ans_list) and len(numbers)==n):
                return 1
        return 0
    else:
        return solution_C(stuAns, None, ans_list, None)

    
# 针对应用题中，设未知数解方程，能够处理多个解的情况，
#能够处理在某些答案后写"舍",写了”舍“字的答案被忽略
def solution_H(stuAns, keywords, ans_list, wrong_lst):
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
    real_answer_lst = list(set([x for x in lst if '舍' not in x]))
    if(len(real_answer_lst)>0):
        return int(ans_list == real_answer_lst)
    else:
        for ans in ans_list:
            if(ans in stuAns):
                return 1
        return 0
    

    
    
# ans_list = [['t-9', '-9+t'], ['15-4t', '-4t+15']]
#适用于多个答案，每个答案多个写法的情况
# 对于高频数字，例如单个数字，调用match_single_digit来匹配
# 每种写法任意匹配到一个就行，多个答案都需要匹配到才行
def solution_I(stuAns, keywords, ans_list, wrong_lst):
    #stuAns = ''.join([x for x in stuAns.split('\n') if ('x=' in x or 'y=' in x)])
    flag = [0 for _ in range(len(ans_list))]
    i = 0
    for answers in ans_list:
        for any_answer in answers:
            if(any_answer.isdigit()):
                #if(any_answer in extract_number(stuAns)):
                if(match_single_digit(stuAns, any_answer)):
                    flag[i] = 1
            else:
                if(any_answer in stuAns):
                    flag[i] = 1
        i += 1
    return int(sum(flag) == len(flag))


# ans_list = [['t-9', '-9+t'], ['15-4t', '-4t+15']]
#适用于多个答案，每个答案多个写法的情况
# 每种写法任意匹配到一个就行，多个答案都需要匹配到才行
# def solution_I(stuAns, keywords, ans_list, wrong_lst):
#     #stuAns = ''.join([x for x in stuAns.split('\n') if ('x=' in x or 'y=' in x)])
#     flag = [0 for _ in range(len(ans_list))]
#     i = 0
#     for answers in ans_list:
#         for any_answer in answers:
#             if(any_answer.isdigit()):
#                 if(any_answer in extract_number(stuAns)):
#                     flag[i] = 1
#             else:
#                 if(any_answer in stuAns):
#                     flag[i] = 1
#         i += 1
#     return int(sum(flag) == len(flag))
 
    
# 一题多问的第一问
# ans_list = [['t-9', '-9+t'], ['15-4t', '-4t+15']]
#适用于多个答案，每个答案多个写法的情况
# 每种写法任意匹配到一个就行，多个答案都需要匹配到才行
def solution_multiple_question_1(stuAns, keywords, ans_list, wrong_lst):
    if('⑵' in stuAns):
        stuAns = stuAns.split('⑵')[0]
    elif('(2)' in stuAns):
        stuAns = stuAns.split('(2)')[0]
    else:
        stuAns = stuAns
    flag = [0 for _ in range(len(ans_list))]
    i = 0
    for answers in ans_list:
        for any_answer in answers:
            if(any_answer in stuAns):
                flag[i] = 1
        i += 1
    return int(sum(flag) == len(flag))

# 一题多问的第二问
# ans_list = [['t-9', '-9+t'], ['15-4t', '-4t+15']]
#适用于多个答案，每个答案多个写法的情况
# 每种写法任意匹配到一个就行，多个答案都需要匹配到才行
def solution_multiple_question_2(stuAns, keywords, ans_list, wrong_lst):
    if('⑵' in stuAns):
        stuAns = stuAns.split('⑵')[1].split('⑵')[0]
    elif('(2)' in stuAns):
        stuAns = stuAns.split('(2)')[1].split('(3)')[0]
    else:
        stuAns = stuAns
    flag = [0 for _ in range(len(ans_list))]
    i = 0
    for answers in ans_list:
        for any_answer in answers:
            if(any_answer in stuAns):
                flag[i] = 1
        i += 1
    return int(sum(flag) == len(flag))


#排序题目策略
# ans_list = ['{{1}^{3}}+{{2}^{3}}+{{3}^{3}}+\cdots +{{100}^{3}}>{{\left( -5000\right)}^{2}}']
def solution_judge_order(stuAns, keywords, ans_list, wrong_lst):
    def helper(list1, ref_list): # helper函数遍历list1的每一个item，并从ref_list中找到和item最相似的，放到foo数组里
        if(len(list1) != len(ref_list)): 
            # print('元素数目不同')
            return 0
        foo  = []
        for item in list1:
            max_sim = -1
            item_star = ''
            for ref in ref_list:
                sim = fuzz.ratio(item, ref)
                if(sim > max_sim):
                    max_sim = sim
                    item_star = ref
            foo.append(item_star)
        return foo == ref_list
    
    lines = stuAns.split('\n')
    for answer in ans_list:
        if('>' in answer):
            # ordered_items为标答的升序排列
            ordered_items = answer.split('>')[::-1] #统一升序
        elif('<' in answer):
            ordered_items = answer.split('<')
        else:
            # print('标答中不包含大于小于号！标答有误！')
            return 1
        
        for line in lines:
            if('>' in line):
                #ordered_items_stu为学生回答的升序排列
                ordered_items_stu = line.split('>')[::-1] #统一升序
                if(helper(ordered_items_stu, ordered_items)==1): #调用helper比较二者是否同顺序
                    return 1
            elif('<' in line):
                ordered_items_stu = line.split('<')
                if(helper(ordered_items_stu, ordered_items)==1): #调用helper比较二者是否同顺序
                    return 1
            else:
                continue
    return 0

# 适用于匹配一般复杂的latex公式，公式只包含一项，例如标答={{\left( a-1 \right)}^{2}}{{\left( b-1 \right)}^{2}}
def solution_match_formula_single(stuAns, keywords, ans_list, wrong_lst):
    stuAns = clean_formula(stuAns)
    for ans in ans_list:
        ans = clean_formula(ans)
        if(ans in stuAns):
            return 1
    return 0


# 逐行匹配
# 适用于匹配较复杂的latex公式，且公式包含多个项，例如标答= '8{{a}^{2}}-12ab-9{{b}^{2}}' 包含3项，需要全部匹配到
def solution_match_formula_multiple(stuAns, keywords, ans_list, wrong_lst):
    ans_list = [clean_formula(x) for x in ans_list]
    stuAns = clean_formula(stuAns)
    for line in stuAns.split('\n'):
        is_correct = [0 for _ in range(len(ans_list))]
        for i, ans in enumerate(ans_list):
            if(ans in line):
                is_correct[i] = 1
        if(sum(is_correct)==len(is_correct)):
            return 1
    return 0

 #硬匹配方法，适用于答案格式固定，不易混的情况
#ans_list 为答案的多种写法，匹配到任意一个即算正确
def solution_match(stuAns, keywords, ans_list, wrong_lst):
    stuAns = union_symbol(stuAns)
    stuAns = stuAns.replace(' ','')
    if wrong_lst != None:
        for i in wrong_lst:
            if i in stuAns:
                return 0
    for i in ans_list:
        if i in stuAns:
            return 1
    return 0


#适用于计算题求值题，一般为单一数字或表达式结果
#可能有多种写法，如a+b，b+a，或假分数、小数
#命中一种写法即为正确
def solution_cal(stuAns, ans_list,wrong_list,keywords=None):
    stuAns = stuAns.split('=')[-1]
    stuAns = union_symbol(stuAns).replace(' ','')
    if wrong_list != None:
        for num in wrong_list:
            if num in stuAns:
                return 0

    for num in ans_list:
        if num in stuAns:
            return 1
    return 0

#适用于求值题有多个答案的情况
#命中全部回答才算正确
#ans_list 可以为一个或多个数，必须全部答对才算正确
#wrong_list 为错误答案，出现即为错误
#keywords 为定位答案的关键词，一般包括“答”“是”等
def solution_calMutil(stuAns, ans_list, wrong_list, keywords):
    if keywords != None:
        lines = stuAns.replace(' ','').split('\n')
        answer_line =[]
        other_line = []
        
        for line in lines:
            if any(kw in line for kw in keywords):
                answer_line.append(line.split('=')[-1])
            else:
                other_line.append(line)
        if (len(answer_line)>0):
            stuAns = ''.join(answer_line)
    if wrong_list != None:
        if (any(ans in stuAns for ans in wrong_list )):
            return 0
    for ans in ans_list:
        if(all(ans in stuAns for ans in ans_list )):
            return 1
    return 0



#适用于答案是整数的题
#ans_list 可以为一个或多个整数，必须全部答对才算正确
#wrong_list 为错误答案，出现即为错误
#keywords 为定位答案的关键词，一般包括“答”“是”等
# def solution_integer(stuAns, ans_list, keywords, wrong_list):
#     lines = stuAns.replace(' ','').split('\n')
#     answer_line =[]
#     other_line = []
        
#     for line in lines:
#         if any(kw in line for kw in keywords):
#             answer_line.append(line)
#         else:
#             other_line.append(line)
    
#     numbers=[]
#     if(len(answer_line)!= 0):
#         for line in answer_line:
#             line = union_symbol(line)
#             number = extractNumbers(line)
#             numbers.extend(number)
#        # if len(numbers) != len(ans_list):
#           #  return 0
#     else:
#         for line in other_line:
#             line = union_symbol(line)
#             number = extractNumbers(line)
#             numbers.extend(number)
   
#     if wrong_list != None:
#         if any(ans in numbers for ans in wrong_list):
#             return 0
        
#     if all(ans in numbers for ans in ans_list):
#         return 1
#     return 0