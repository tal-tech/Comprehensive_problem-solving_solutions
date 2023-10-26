#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import Levenshtein


################################################################
# 以下为批答案所需
################################################################

# 定位答案行，
#n是标答的数字的个数，例如标答=5、10、15、20，则n=4
#stuAns为学生回答，线上是OCR输出的文本
#keywords为定位答案行的关键字，一般包含“答”字，具体根据题干的问法改变
def locateAnswerLine_new(stuAns, keywords, n):
    lines = stuAns.split('\n') 
    answerLines, otherLines = [], []
    
    for i in range(len(lines)):
        line = lines[i]
        next_line = lines[i+1] if (i+1)<len(lines) else ''
        isAnswer = False
        m = len(extractNumbers(line))
        for k in keywords:
            if(k in line):
                if(m>=n):
                    answerLines.append(line)
                else:
                    line += next_line
                    answerLines.append(line)
                isAnswer = True
                break
        if(isAnswer==False):
            otherLines.append(line)
    return answerLines, otherLines

# 仅根据关键词寻找答案行
def locateAnswerLine(stuAns, keywords):
    if(keywords==None or keywords==[]):
        keywords = ['答']
    lines = stuAns.split('\n') 
    answerLines = []
    
    for i in range(len(lines)):
        line = lines[i]
        for k in keywords:
            if(k in line):
                answerLines.append(line)
                break
    return answerLines


# 从字符串中抽取出数字，返回为数字字符串的list，无法提取负号，例如-10会被提取成10
def extractNumbers(line):
    numbers = re.findall(r"\d+\.?\d*", line)
    numbers = [re.sub(r'[^\w\s]','', n) for n in numbers] 
    return numbers

# 迭代后的抽取数字，支持int float 正负号
def extract_number(s):
    return re.findall('-?\d+(?:\.\d+)?', s)

def getKeyIdx(doc, query):
    w = len(query)
    for st in range(len(doc)-w+1):
        if(doc[st: st+w]==query):
            return st
    return 0

# 统一latex写法
def union_symbol(strip_raw_text):

    raw_list = ["\\right","\\left","\\rm","\\leqslant","\\mathsf","\\underbrace",
                "\\geqslant","\\bigstar","\\quad","\\hline","\\dfrac","\\triangle","\\Delta",
                "\\Rightarrow","\\rightarrow","\\alpha","\\beta","\\rho","\\mu","\\theta",
                "\\times","\\div","\\pi","\\angle","{}^\\circ","^\\circ","^{\\circ}","\\cdots","\\cdot",
                "\\ldots","\\pm","\\because","\\therefore","\\neq","\\geq","\\leq","\\equiv",
                "\\approx","\\Square","\\square","\\max","\\min","\\cos","\\sin","\\tan",
                "\\%","\\_","\\downarrow","\\uparrow","\\ast","\\oplus","\\sim","\\bmod",
                "\\longrightarrow","\\Downarrow","\\Uparrow","\\arrow","\\mathrm"
                ]
    replace_list=[" "," "," "," "," "," ",
                  " "," "," "," ","\\frac","△","△",
                  "→","→","α","β","ρ","μ","θ",
                  "×","÷","π","∠","°","°","°","···","·",
                  "...","±","∵","∴","≠","≥","≤","≡",
                  "≈","□","□","max","min","cos","sin","tan",
                  "%","_","↓","↑","*","⊕","∼","mod",
                  "→","↓","↑","→",""
                  ]

    for tmp_sym, tar_sym in zip(raw_list,replace_list):
        if tmp_sym in strip_raw_text:
            strip_raw_text = strip_raw_text.replace(tmp_sym,tar_sym)
    return strip_raw_text


def clean(x):
    '''
    x:
        type: str
    '''
    if(len(x)==0):
        return x
    x = "".join(x.split())
    x = x.replace('\quad',',')
    x = x.replace('\div','/')
    x = x.replace('\rightarrow','->')
    x = x.replace('\\rightarrow','->')
    x = x.replace('Rightarrow','->')
    x = x.replace('\geq','>=')
    x = x.replace('\left(','(')
    x = x.replace('\right)',')')
    x = x.replace('\\right)',')')
    x = x.replace('\times','*')
    x = x.replace('\\times','*')
    x = x.replace('\equiv','=')
    x = x.replace('\bmod','mod')
    x = x.replace('\geqslant','>=')
    x = x.replace('\cdots','...')
    x = x.replace('\cdot','.')
    x = x.replace('\neq','!=')
    x = x.replace('\operatorname','')
    x = x.replace('\leq','<=')
    x = x.replace('\\dfrac', '\\frac')
    x = x.replace(' ', '')
    x = x.replace('．', '')
    x = x.replace('、', ',')
    x = x.replace('，', ',')
    x = x.replace('或', ',')
    
    if(x[-1]=='.'):
        x = x[:-1]
    return x


def match_multiple_single_digit(string, x_lst):
    flag = [0 for _ in range(len(x_lst))]
    for i, x in enumerate(x_lst):
        if(not x.isdigit()):
            flag[i] = 1 if x in string else 0
        else:
            if(match_single_digit(string, x)==1):
                flag[i] = 1
    return int(sum(flag)==len(flag))

def is_26_english_char(s):
    return s >= 'a' and s <='z' or s >= 'A' and s <='Z'

def match_single_digit(string, x):
    def helper(s):
        return s in ['+', '-', '*', '/', '{', '}'] or is_26_english_char(s) or s.isdigit()
    
    assert x.isdigit() == True
    num_digit = len(x)
    for st in range(len(string)):
        ed = min(st+num_digit, len(string))
        string_slice = string[st : ed]
        if(string_slice == x):
            #print(st, ed)
            pre_char = string[st-1] if (st-1)>=0 else ''
            aft_char = string[ed] if ed <= len(string)-1 else '' 
            #print(pre_char, aft_char)
            #if(aft_char=='' or (pre_char in [',', '，', '、', '=', ''] and aft_char in [',', '，', '、'])):
            if(aft_char in [''] or (not helper(pre_char) and not helper(aft_char))):
                return 1
    return 0


def clean_formula(s):
    s = s.replace('\left', '')
    s = s.replace('\\left', '')
    s = s.replace('\right', '')
    s = s.replace('\\right', '')
    s = s.replace('{', '')
    s = s.replace('}', '')
    s = s.replace(' ', '')
    return s

################################################################
# 以下为批过程所需
################################################################
def calculate_distance(procedure_list, stu_answer_row):
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
        tmp.append(Levenshtein.distance(u'{}'.format(procedure),u'{}'.format(stu_answer_row)))
    min_dist = min(tmp)
    return min_dist

def extract_float_number(text):
    '''
    text:
        type: str
        sample: '1+1=2'
    '''
    return re.findall(r"\d+\.?\d*",text)