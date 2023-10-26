#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
import math
from procedure_model.src.procedure_utils import *

sequence_1 = list(map(lambda x:(chr(x)), range(ord('A'), ord('Z')+1)))
sequence_2 = list(map(lambda x:(chr(x)), range(ord('a'), ord('z')+1)))
sequence = sequence_1 + sequence_2
num_sequence = ['1','2','3','4','5','6','7','8','9']
def extend_overline(procedure):
    new_list = []
    gai_dict = {}
    #find_inner = re.findall(r'(\\overline{[\d+]*[a-zA-Z]*[\d+]*})',procedure)
    find_inner = re.findall(r'(\\overline\s*{[A-Za-z0-9]+})',procedure)
    if len(find_inner)>0:
        for inner in find_inner:
            true_inner = re.findall(r'\\overline\s*\{([A-Za-z0-9]+)\}',inner)[0]
            start = 0
            new_inner = []
            for wei in true_inner[::-1]:
                index_num = str(int(math.pow(10,start)))
                if wei == '0':
                    pass
                elif wei in num_sequence:
                    wei = int(wei)*int(index_num)
                    new_inner.append(str(wei))
                else:
                    if index_num == '1':
                        new_inner.append(wei)
                    else:
                        new_inner.append(index_num+wei)
                start+=1
            gai_dict[inner] = '('+'+'.join(new_inner[::-1])+')'
            new_list.append(procedure.replace(inner,gai_dict[inner]))
        new_pro = copy.copy(procedure)
        for i in gai_dict:
            new_pro =new_pro.replace(i,gai_dict[i])
        new_list.append(new_pro)
    newnew_list = copy.copy(new_list)
    for i in new_list:
        num_pre = re.findall(r'[\d]+\s*\([0-9a-zA-Z+]*\)',i)
        num_aft = re.findall(r'\([0-9a-zA-Z+]*\)\s*\[\d]+',i)
        this_new_i = {}
        for pre in num_pre:
            num_cheng = re.findall(r'\d+',pre[:pre.find('(')])[0]
            #num_cheng = int(pre[:pre.find('(')])
            num_kuohao = pre[pre.find('(')+1:pre.find(')')]
            inner_element = re.findall(r'[\da-zA-Z]+',num_kuohao)
            new_inner_element = []
            for ele in inner_element:
                num_ele = re.findall(r'[\d]+',ele)
                if num_ele:
                    new_inner_element.append(ele.replace(num_ele[0],str(int(num_ele[0])*int(num_cheng))))
                else:
                    new_inner_element.append(str(num_cheng)+ele)
            if new_inner_element:
                new_pre = '+'.join(new_inner_element)
            this_new_i[pre] = new_pre
        for aft in num_aft:
            num_cheng = re.findall(r'\d+',aft[:aft.find('(')])[0]
            #num_cheng = int(aft[aft.find(')')+1:])
            num_kuohao = aft[aft.find('(')+1:aft.find(')')]
            inner_element = re.findall(r'[\da-zA-Z]+',num_kuohao)
            new_inner_element = []
            for ele in inner_element:
                num_ele = re.findall(r'[\d]+',ele)
                if num_ele:
                    new_inner_element.append(ele.replace(num_ele[0],str(int(num_ele[0])*int(num_cheng))))
                else:
                    new_inner_element.append(str(num_cheng)+ele)
            if new_inner_element:
                new_aft = '+'.join(new_inner_element)
            this_new_i[aft] = new_aft
        c = copy.copy(i)
        for cc in this_new_i:
            c = c.replace(cc,this_new_i[cc])
        c = c.replace('(','')
        c = c.replace(')','')
        newnew_list.append(c)
    newnew_list.append(procedure)
    return list(set(newnew_list))

def extend_overline_pro_list(procedure_list):
    all_list = []
    for i in procedure_list:
        if 'overline' in i:
            extended = extend_overline(i)
            all_list += extended
        else:
            all_list.append(i)
    unique_list = list(set(all_list))
    return unique_list

def extend_B007(procedure_list):
    all_list = []
    for i in procedure_list:
        if 'overline' in i:
            extended = extend_overline(i)
            all_list += extended
        else:
            all_list.append(i)
    unique_list = list(set(all_list))
    extend_this_equation = {}
    for equation in unique_list:
        if ('(' in equation) & (')' in equation):
            pass
        else:
            equation = equation.replace(' ','')
            if '=' in equation:
                left_equal = equation[:equation.find('=')]
                right_equal = equation[equation.find('=')+1:]
                if left_equal[0] != '-':
                    left_equation = '+' + left_equal
                if right_equal[0] != '-':
                    right_equation = '+' + right_equal
                left_element = re.findall(r'[\+\-]+[0-9a-zA-Z]+(?!\\)',left_equation)
                right_element = re.findall(r'[\+\-]+[0-9a-zA-Z]+(?!\\)',right_equation)
                left_overline = re.findall(r'[\d]*\\overline{[a-zA-Z0-9]+}',left_equation)
                right_overline = re.findall(r'[\d]*\\overline{[a-zA-Z0-9]+}',right_equation)
                for ele in right_element:
                    if '+' in ele:
                        left_element.append(ele.replace('+','-'))
                    else:
                        left_element.append(ele.replace('-','+'))
                this_group = {}
                for ele in left_element:
                    this_ele_eng = re.findall('[a-zA-Z]+',ele)
                    if this_ele_eng:
                        num = ele.replace(this_ele_eng[0],'')
                        if (num == '+') or (num=='-'):
                            num = num + '1'
                        if this_ele_eng[0] in this_group:
                            this_group[this_ele_eng[0]] += num
                        else:
                            this_group[this_ele_eng[0]] = num
                    else:
                        if 'num' in this_group:
                            this_group['num'] += ele
                        else:
                            this_group['num'] = ele
                for i in this_group:
                    this_group[i] = eval(this_group[i])
            this_group_tuozhan = []
            #print(this_group)
            #print(left_element)
            single_right = '+'.join(right_overline)
            single_left = '+'.join(left_overline)
            for i in this_group:
                if this_group[i] == 0:
                    pass
                elif (this_group[i] == 1) & (i in sequence):
                    single_left += '+'+i
                elif this_group[i] >0:
                    yaojia = '+' + str(this_group[i]) +i
                    single_left += yaojia
                elif this_group[i]<0:
                    single_right += '+'+ (str(this_group[i])+i).replace('-','')
            single_left = single_left.replace('num','')
            single_right = single_right.replace('num','')
            if len(single_left)>0:
                if single_left[0] == '+':
                    single_left = single_left[1:]
            if len(single_right)>0:
                if single_right[0] == '+':
                    single_right = single_right[1:]
            single_left, single_right = single_left.replace('num',''), single_right.replace('num','')
            if single_left == '':
                single_left = '0'
            if single_right == '':
                single_right = '0'
            this_group_tuozhan.append(single_left+'='+single_right)
            for i in this_group:
                if i in sequence:
                    copy_element = copy.copy(left_element)
                    for g in left_element:
                        if i in g:
                            copy_element.remove(g)
                    if this_group[i] == 1:
                        copy_element.append('+'+str(i))
                    elif this_group[i] >0:
                        copy_element.append('+'+str(this_group[i])+str(i))
                    elif this_group[i] == 0:
                        pass
                    else:
                        copy_element.append(str(this_group[i])+str(i))
                    mid_right =  '+'.join(right_overline)
                    mid_left = '+'.join(left_overline)
                    for use in copy_element:
                        if '-' in use:
                            mid_right = mid_right+'+'+use.replace('-','')
                        else:
                            mid_left = mid_left+use
                    if (len(mid_left)>0):
                        if mid_left[0] == '+':
                            mid_left = mid_left[1:]
                    if (len(mid_right)>0):
                        if mid_right[0] == '+':
                            mid_right = mid_right[1:]
                    if mid_left == '':
                        mid_left = '0'
                    if mid_right == '':
                        mid_right = '0'
                    this_group_tuozhan.append(mid_left+'='+mid_right)
                else:
                    copy_element = copy.copy(left_element)
                    for g in left_element:
                        if not re.findall(r'[a-zA-Z]',g):
                            copy_element.remove(g)
                    if this_group[i] > 0:
                        this_group_i = '+' + str(this_group[i])
                        copy_element.append(this_group_i)
                    elif this_group[i] == 0:
                        pass
                    else:
                        copy_element.append(str(this_group[i]))
                    mid_right = '+'.join(right_overline)
                    mid_left = '+'.join(left_overline)
                    for use in copy_element:
                        if '-' in use:
                            mid_right = mid_right+'+'+use.replace('-','')
                        else:
                            mid_left = mid_left+use
                    if (len(mid_left)>0):
                        if mid_left[0] == '+':
                            mid_left = mid_left[1:]
                    if (len(mid_right)>0):
                        if mid_right[0] =='+':
                            mid_right = mid_right[1:]
                    if mid_left == '':
                        mid_left = '0'
                    if mid_right == '':
                        mid_right = '0'
                    this_group_tuozhan.append(mid_left+'='+mid_right)
            all_list += this_group_tuozhan
        #print(this_group_tuozhan) 

    return list(set([i.replace(' ','') for i in all_list]))

def extract_number_B007(s):
    xiaoliao = re.findall(r'(\d+[a-zA-Z]*)+', s) + re.findall(r'\\overline{[a-zA-Z0-9]+}',s)
    daliao = re.findall(r'\+[a-zA-Z]+',s)
    heihei = []
    for i in daliao:
        heihei.append(i.replace('+',''))
    return heihei+xiaoliao

def calculate_similarity_B007(procedure_list, stu_answer_row):
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

def rule_overline(stuAns_row, procedure, thres):
    procedure_list = [clean(procedure)]
    #procedure_list = extend_overline_pro_list(procedure_list)
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




