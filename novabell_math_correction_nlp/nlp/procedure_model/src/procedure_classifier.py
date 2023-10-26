#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy 
import math
from procedure_model.src.procedure_utils import *

def is_rotate_number(procedure):
    pro = copy.copy(procedure)
    xiaoshudian_hou = re.findall(r"(?<=\.)\d+\.?\d*", pro)
    if any(len(x) > 6 for x in xiaoshudian_hou):
        return True
    else:
        return False
def is_gongyueshu(procedure):
    pro = copy.copy(procedure)
    pro = clean(pro)
    if ('(' in pro) & (')' in pro):
        temps = re.findall(r'\(\d+[\s\S]*\d+\)', pro)
        for temp in temps:
            if ',' in temp:
                res = eval(temp)
                nums = re.findall(r'\d+', temp)
                if len(res) == len(nums):  # 公倍数
                    return True
                else:
                    return False
def is_gongbeishu(procedure):
    pro = copy.copy(procedure)
    pro = clean(pro)
    if ('[' in pro) & (']' in pro):
        temps = re.findall(r'\[\d+[\s\S]*\d+\]',pro)
        for temp in temps:
            if ',' in temp:
                res = eval(temp)
                nums = re.findall(r'\d+',temp)
                if len(res) == len(nums):
                    return True
                else:
                    return False
                
def is_zhongwen_jia_zhongwen(procedure):
    pro = clean(procedure)
    fanshi = re.findall(r'[\u4E00-\u9FA5]+\s*\+\s*[\u4E00-\u9FA5]+\s*\=\s*[\u4E00-\u9FA5]+',pro)
    if fanshi:
        return True
    else:
        return False
    
def is_yushu(procedure):
    if '=' in procedure:
        if any(x in procedure[procedure.find('='):] for x in ['cdots','dots','余']):
            return True
        else:
            return False
    else:
        if ('div' in procedure) & ('余' in procedure):
            return True
        else:
            return False
        
def is_zimusuanshi(procedure):
    pro = clean(procedure)
    fanshi_1 = re.findall(r'[a-zA-Z]+\*\d+(\+|\-|\*\\/)\d+\=\d+',pro)
    fanshi_2 = re.findall(r'\d+[a-zA-Z]+(\+|\-|\*|\/)\d+\=\d+',pro)
    fanshi_3 = re.findall(r'\d+\*[a-zA-Z]+(\+|\-|\*|\/)\d+\=\d+',pro)
    if fanshi_1 or fanshi_2 or fanshi_3:
        return True
    else:
        return False
    
def is_dengcha(procedure):
    if ('=' in procedure) & (any(x in procedure[procedure.find('='):] for x in ['cdots','dots','余'])):
        return False
    elif any(x in procedure for x in ['cdots','dots','...']):
        all_num = re.findall(r'\d+',procedure)
        cdot_index = procedure.find('dots')
        pre_number = re.findall(r'\d+',procedure[:cdot_index])
        if len(pre_number) <3:
            return False
        head = eval(pre_number[0])
        cha = eval(pre_number[1])-eval(pre_number[0])
        if (eval(pre_number[1])-eval(pre_number[0])) == (eval(pre_number[2])-eval(pre_number[1])):
            return True
        else:
            return False
def is_suanshi(procedure):
    pro = copy.copy(procedure)
    pro = clean(pro)
    if '=' in pro:
        left_part = pro[:pro.find('=')]
        right_part = pro[pro.find('=')+1:]
        try:
            if int(eval(left_part)) == int(eval(right_part)):
                return True
            else:
                return False
        except:
            return False

def is_pingfang_suanshi(procedure):
    pro = copy.copy(procedure)
    pro = clean(pro)
    if ('=' in pro) & ('^' in pro):
        fanshi = re.findall(r'[\d]+\s*\^\s*[\d]+',pro)
        this_change = {}
        for fan in fanshi:
            fanhou = re.findall(r'(?<=\^)\d+',fan)[0]
            fanqian = re.findall(r'[\d]+(?=\^)',fan)[0]
            this_change[fan] = ((fanqian+'*')*int(fanhou))[:-1]
        for i in this_change:
            pro = pro.replace(i, this_change[i])
        left_part = pro[:pro.find('=')]
        right_part = pro[pro.find('=')+1:]
        try:
            if int(eval(left_part)) == int(eval(right_part)):
                return True
            else:
                return False
        except:
            return False
    else:
        return False

def is_pingfang_budengshi(procedure):
    pro = clean(procedure)
    fanshi_1 = re.findall(r'\d+\^\d+(?:\<|\>|\<=|\>=)\d+',pro)
    fanshi_2 = re.findall(r'\d+(?:\<|\>|\<=|\>=)\d+\^\d+',pro)
    if fanshi_1 or fanshi_2:
        return True
    else:
        return False

def is_jiefangcheng(procedure):
    pro = copy.copy(procedure)
    pro = clean(pro)
    if (len(re.findall(r'[x|y|z]',pro)) > 0) &('\overline' not in procedure):
        return True
    else:
        return False
    
def is_chinese(procedure):
    pro = clean(procedure)
    chi_num = re.findall(r'[\u4E00-\u9FA5]',pro)
    if len(chi_num)/len(pro) > 0.6:
        return True
    else:
        return False

def is_putong_shulie(procedure):
    pro = clean(procedure)
    chi = re.findall(r'[\u4E00-\u9FA5]+',pro)
    operator = re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', pro)
    if (not operator) & (not chi):
        if pingfang_shulie(procedure):
            return False
        else:
            return True
    else:
        return False

def is_shuzi_zhongwen(procedure):
    pro = clean(procedure)
    fanshi = re.findall(r'[\u4E00-\u9FA5]+\*[\d]+',pro)
    if fanshi:
        return True
    else:
        return False


def is_pingfang_shulie(procedure):
    pro = clean(procedure)
    chi = re.findall(r'[\u4E00-\u9FA5]+',pro)
    operator = re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', pro)
    if (not operator) & (not chi):
        if pingfang_shulie(procedure):
            return True
    else:
        return False

def is_overline(procedure):
    pro = clean(procedure)
    if '\overline' in pro:
        return True
    else:
        return False

# 玮萍
def is_all_overline(procedure):
    pro = clean(procedure)
    if re.match(r'^(?:\\overline\s*\{[a-zA-Z0-9]+\}[\+,\-\s*、;；，]\s*)+\\overline\s*\{[a-zA-Z0-9]+\}$', pro):
        return True
    return False

#玮萍
def is_shuzi_zimu(procedure):
    pro = clean(procedure)
    matches = re.findall(r'[1-9][0-9]*\((?:\d*[a-zA-Z]+[\+-])+\d*[a-zA-Z]+\)', pro)
    matches += re.findall(r'\d+\s*(?:\*|\/)\s*\((?:\d*[a-zA-Z]+[\+-])+\d*[a-zA-Z]+\)', pro)
    matches += re.findall(r'\((?:\d*[a-zA-Z]+[\+-])+\d*[a-zA-Z]+\)\s*(?:\*|\/)\s*\d+', pro)
    if len(matches) > 0:
        return True
    return False

def is_ziweishu(procedure):
    pro = clean(procedure)
    fanshi = re.findall(r'[\u4E00-\u9FA5]+为\d+',pro)
    if fanshi:
        if fanshi[0] == pro:
            return True
        else:
            return False
    else:
        return False

def pingfang_shulie(procedure):
    procedure_clean = clean(procedure)
    pro = copy.copy(procedure)
    new_list = []
    new_list.append(procedure_clean)
    chi = re.findall('[\u4E00-\u9FA5]+',pro)
    operator =re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', pro)
    if (not chi) and (not operator):
        num = re.findall(r'\d+',pro)
        new_num = []
        for i in num:
            genhao = math.sqrt(eval(i))
            if genhao - int(genhao) == 0:
                new_num.append(str(int(genhao))+'^2')
        if len(new_num) == len(num):
            return True
    return False

def procedure_classify(procedure):
    '''
    Input:
        question_content(str):
            一个大于1的自然数除17、45、97的余数相同，这个自然数最大可能是几？
        procedure_list(str):
            97-45=52\n45-17=28\n\left(52,28\right)=4\n
    Output:
        the classification of this question, include:
        1，循环小数
        2, 公约数
        3，公倍数
        4，解方程
        5，除法余数
        6，等差数列
        7，其他
    '''
    if is_rotate_number(procedure):
        return '循环小数'
    elif is_shuzi_zimu(procedure):
        return '数字跟字母运算'
    elif is_gongyueshu(procedure):
        return '公约数'
    elif is_gongbeishu(procedure):
        return '公倍数'
    elif is_yushu(procedure):
        return '除法余数'
    elif is_jiefangcheng(procedure):
        return '解方程'
    elif is_dengcha(procedure):
        return '等差数列'
    elif is_suanshi(procedure):
        return '普通算式'
    elif is_pingfang_suanshi(procedure):
        return '平方算式'
    elif is_pingfang_budengshi(procedure):
        return '平方不等式'
    elif is_zimusuanshi(procedure):
        return '字母算式'
    elif is_ziweishu(procedure):
        return '字为数'
    elif is_zhongwen_jia_zhongwen(procedure):
        return '中文+中文'
    elif is_chinese(procedure):
        return '中文描述'
    elif is_putong_shulie(procedure):
        return '普通数列'
    elif is_pingfang_shulie(procedure):
        return '平方数列'
    elif is_all_overline(procedure):
        return '全部overline'
    elif is_overline(procedure):
        return 'overline相关'
    elif is_shuzi_zhongwen(procedure):
        return '数字中文'
    else:
        return '其他'

if __name__ == "__main__":
    procedure = '1+3+5+7+\\cdots+13+4'
    # print(procedure_classify(procedure))





