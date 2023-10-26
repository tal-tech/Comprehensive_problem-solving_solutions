import cv2
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import Config as config
import sympy
import math
from exception import EquationInvalidError

class Result(object):
    correct, incorrect, problem = range(3)

def image_normalization(image):
    if image.shape[0] != 32:
        image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * 32), 32))
    if image.shape[1] < 10:
        image = cv2.resize(image, (10, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255 * 2 - 1
    return image

def PILtoOpenCV(img_PIL):
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV

def OpenCVtoPIL(img_OpenCV):
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))
    return img_PIL

def boundingBox(Bbox1, Bbox2):
    top = min(Bbox1.top, Bbox2.top)
    bottom = max(Bbox1.bottom, Bbox2.bottom)
    left = min(Bbox1.left, Bbox2.left)
    right = max(Bbox1.right, Bbox2.right)
    return left, top, right, bottom

def fraction_cvt(output_list):
    for i in range(len(output_list)):
        output_list[i] = output_list[i].replace('{(', '{')
        output_list[i] = output_list[i].replace(')}', '}')
        output_list[i] = output_list[i].replace(')(', '|')
        output_list[i] = output_list[i].replace(' ', '')
    return output_list

'''
判题用utils
'''
# kk:交叉判题模块，可用于（带题干的竖式）和（脱式的二次判题）
def crossJudge(stemList, ansList, yushuList = [], type = 'HS'):
    yushuLen = len(yushuList)
    for stem in stemList:
        for ans in ansList:
            equation = stem + '=' + ans
            return_state, return_output = judge(equation, type = type)
            if return_state == Result.correct:# kk:先判，不对再加余数
                return return_state, return_output

            if '÷' in stem and yushuLen > 0:
                for yushu in yushuList:
                    try:
                        if float(yushu) < float(stem.split('÷')[1]):
                            equation = equation + '***' + yushu

                            return_state, return_output = judge(equation, type = type)
                            if return_state == Result.correct:
                                return return_state, return_output
                    except:
                        continue
    stem = stemList[0]
    ans = ansList[0]
    equation = stem + '=' + ans
    if '÷' in stem and yushuLen > 0:
        yushu = yushuList[0]
        if yushu != '0' and float(yushu) < float(stem.split('÷')[1]):
            equation = equation + '***' + yushu
    return Result.incorrect, equation


def get_legal_level(text):
    #1+1=2 合法 3
    #1+1=  合法 2
    #1+1   合法 1
    state, _ = judge(text)
    if state != Result.problem:
        return 3
    else:
        text_temp = text + '1'
        state, _ = judge(text_temp)
        if state != Result.problem:
            return 2
        else:
            text_temp = text + '= 1'
            state, _ = judge(text_temp)
            if state != Result.problem:
                return 1
            else:
                return 0


def isdigit(text):
    if text == '':
        return False
    if text[0] == '.' or text[-1] == '.':
        return False
    if text[0] == '-' and len(text) > 1:
        text = text[1:]
    return text.replace(".", "").isdigit()


def islegal(text, level = 1):
    #1+1=2 合法 3
    #1+1=  合法 2
    #1+1   合法 1
    if text == '':
        return False
    if '=' in text:
        equation_left = text.split('=')[0]
        equation_right = text.split('=')[-1]
        if level == 3:
            if equation_right == '' or equation_left == '':
                return False
        elif level == 2:
            if equation_right == '' and equation_left == '':
                return False
    if level == 3:
        state, _ = judge(text)
        if state != Result.problem:
            return True
    elif level == 2:
        text_temp = text + '1'
        state, _ = judge(text_temp)
        if state != Result.problem:
            return True
    elif level == 1:
        text_temp = text + '= 1'
        state, _ = judge(text_temp)
        if state != Result.problem:
            return True
    return False



def judge(text, type = "HS"):


    text = delete_draft(text)
    text_raw = '%s' % (text)
    if '□' in text:
        return Result.incorrect, text_raw
        # text1 = text.replace('□', '=')
        # text2 = text.replace('□', '1')
        # if islegal(text1, level=3) or islegal(text2, level=3):
        #     return Result.incorrect, text_raw
        # else:
        #     return Result.problem, None

    # kk: 删掉 (1.4), (周), (5)这种无意义的括号

    text = delete_brackets(text)

    # kk: 带单位的带余除法: 37 / 2 = 5(周)******2(天)
    if '*' in text or '~' in text:
        text = delete_unit(text)

    # kk: 处理单位
    if has_unit(text):
        text = '(' + text + ')'
        ops = ['+', '-', '=', '>', '<']
        for op in ops:
            text = text.replace(op, ')' + op + '(')
        text, flag = unit_convert(text)  # kk: 横式单位转换
        # kk: 转换一次单位后单位数大于1,则说明该横式为类似于'1年=1km'这种无效式子,直接判错
        # 否则认为是普通的带单位横式,去掉单位
        unit_num = get_unit_num(text)
        if unit_num > 1:
            return Result.incorrect, text_raw
        else:
            text = delete_unit(text)

    text = replace_question_num(text)  # 删除题号

    if '|' in text:
        text = fraction_replace(text)
    text = text.replace(':', '÷')  # 求比例替换
    text = text.replace('[', '(')  # 中括号替换
    text = text.replace(']', ')')
    text = text.replace('%', '÷100')
    if '()' in text and (set('><≈=') & set(text)):
        return Result.incorrect, text_raw
    if '××' in text:
        return Result.problem, None
    if set('><') & set(text):
        return judge_inequation(text), text_raw
    elif '*' in text or '~' in text:
        return judge_residue(text), text_raw
    elif '≈' in text:
        return judge_apequal(text), text_raw
        #kk: 约等于判题策略有待完善, 目前策略为当两边之差小于两边最小数的0.1倍时就判对
    else:
        return judge_equation(text, type), text_raw


def no_chinese(string):
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return False
    return True

def has_unit(string):
    if set(config.UNIT) & set(string):
        return True
    return False


def get_unit_num(string):
    units = re.findall('[\u4e00-\u9fa5]+', string)
    units_list = []
    for unit in units:
        if unit not in units_list:
            units_list.append(unit)
    return len(units_list)

def unit_convert(string):
    flag = False
    for measure in config.UNIT_CONVER_LIST:
        replace_list = []
        if not flag:
            for unit in measure:
                match_str = re.search('[^\u4e00-\u9fa5]'+unit, string)
                if match_str:
                    match_str = match_str.group()[1:]
                    replace_list.append(match_str)
                    # string = string.replace(match_str, '×' + str(measure[unit]) + '+')
                    # string = string.replace(match_str, match_str.replace(unit, '×'+str(config.UNIT_CONVER[unit])))
        if len(replace_list) > 1:
            flag = True
            for replace_str in replace_list:
                string = string.replace(replace_str, '×' + str(measure[replace_str]) + '+')

    string = string.replace('+)', ')')
    # string = string.replace('+=', '=')
    # string = string.replace('+>', '>')
    # string = string.replace('+<', '<')
    # string = string.replace('+-', '-')
    if string[-1] == '+':
        string = string[:-1]
    if string[-2:] == '+)':
        string = string[:-2]
    return string, flag


def delete_draft(text):
    units = re.findall('□{\d+\.?\d*}', text)
    for unit in units:
        text = text.replace(unit, '')
    return text

def delete_unit(text):
    units = re.findall('[\u4e00-\u9fa5]+', text)
    for unit in units:
        text = text.replace(unit, '')
    return text


def delete_brackets(text):
    brackets = re.findall('\(\d+\.?\d*\)', text)
    for bracket in brackets:
        bracket_change = bracket.replace('(','')
        bracket_change = bracket_change.replace(')','')
        text = text.replace(bracket, bracket_change)

    brackets = re.findall('\([\u4e00-\u9fa5]+\)', text)
    for bracket in brackets:
        bracket_change = bracket.replace('(','')
        bracket_change = bracket_change.replace(')','')
        text = text.replace(bracket, bracket_change)
    return text


def fraction_replace(string):
    string = string.replace('|', '/')
    #1{2/3}替换为(1+{2/3})
    fraction_list = re.findall('\d{[^}]*}', string)
    for fraction in fraction_list:
        string = string.replace(fraction,'('+fraction.replace('{','+{')+')')

    #去掉{}，添加（）
    fraction_list = re.findall('{[^}]*}', string)
    for fraction in fraction_list:
        fraction_change = fraction.replace('{','')
        fraction_change = fraction_change.replace('}', '')
        if '/' in fraction_change:
            fraction_change_list = fraction_change.split('/')
            fraction_change = '('+'('+fraction_change_list[0]+')'+'/'+'('+fraction_change_list[1]+')'+')'
        else:
            fraction_change = '('+fraction_change+')'
        string = string.replace(fraction, fraction_change)

    return string

def replace_question_num(text):
    text_list = text.split(')')
    if len(text_list) > 1 and text_list[1] and (text_list[1][0] not in '+-×÷><=*x|'):
        text = ''.join(text_list[1:])
    text = text.replace('○', '')
    return text

def judge_equation(string, type = 'HS'):
    try:
        if ('=' not in string) or (string == '') or (re.search(r'[a-zA-Z]', string)):
            return Result.problem
        else:
            string = string.replace('×', '*')
            string = string.replace('÷', '/')
            equation_left = string.split('=')[0]
            equation_right = string.split('=')[-1]
        if not type == 'HS' and not (re.fullmatch(r'\(\(\d+\)/\(\d+\)\)', equation_right)):
            if (set('+-*/') & set(equation_right)): #and set('+-*/') & set(equation_left):
                #kk:判脱式和解方程,如果右边为一个式子,直接判错
                return Result.incorrect

        if equation_right == '' or equation_left == '':
            return Result.incorrect

        if equation_right[-1] == '(':
            equation_right = equation_right[:-1]

        left = eval(equation_left)
        right = eval(equation_right)
        diff = abs(left - right)
        if diff < 1e-8:
            return Result.correct
        else:
            return Result.incorrect
    except:
        return Result.problem


def judge_apequal(string):
    try:
        if re.search(r'[a-zA-Z]', string):
            return Result.problem
        else:
            string = string.replace('×', '*')
            string = string.replace('÷', '/')
            equation_left = string.split('≈')[0]
            equation_right = string.split('≈')[-1]

        if set('+-*/') & set(equation_right) and set('+-*/') & set(equation_left) :
            return Result.incorrect

        if equation_right == '' or equation_left == '':
            return Result.problem

        left = eval(equation_left)
        right = eval(equation_right.split('(')[0])
        if (abs(left - right) < 0.1 * min(abs(left), abs(right))):
            return Result.correct
        else:
            return Result.incorrect
    except:
        return Result.problem


def judge_inequation(string):
    try:
        string = string.replace('×','*')
        string = string.replace('÷','/')
        if '>' in string:
            subString_list = string.split('>')
            if eval(subString_list[0]) > eval(subString_list[1]):
                return Result.correct
            else:
                return Result.incorrect
        else:
            subString_list = string.split('<')
            if eval(subString_list[0]) < eval(subString_list[1]):
                return Result.correct
            else:
                return Result.incorrect
    except:
        return Result.problem

def judge_residue(string):
    try:
        if '=' not in string or '÷' not in string:
            return Result.problem
        else:
            left = string.split('=')[0]
            right = string.split('=')[1]
        if right == '' or left == '':
            return Result.problem
        if '*' in left:
            return Result.problem
        left = left.replace('×', '*')
        left1 = left.replace('÷', '//')
        left2 = left.replace('÷', '%')
        left1 = eval(left1)
        left2 = eval(left2)
        if '*' in right or '~' in right:
            right1 = ''
            right2 = ''
            if '*' in right:
                right1 = right.split('*')[0]
                right2 = right.split('*')[-1]
            if '~' in right:
                right1 = right.split('~')[0]
                right2 = right.split('~')[-1]
            right1 = eval(right1)
            right2 = eval(right2)
            if right1 == int(left1) and right2 == int(left2):
                return Result.correct
            else:
                return Result.incorrect
        else:
            if left2 == 0:
                if left1 == int(right):
                    return Result.correct
                else:
                    return Result.incorrect
            else:
                return Result.problem
    except:
        return Result.problem

def solve_equation(text):
    # 分式符号替换
    if '|' in text:
        text = fraction_replace(text)
    #判断方程是否有问题
    if '=' not in text:
        raise EquationInvalidError('方程不合法，其中不包含等号')
    text = text.replace('%', '÷100')
    text = text.replace(':', '÷')
    text = text.replace('÷', '/')
    text = text.replace('×','*')
    text = text.replace('x(', 'x*(')
    text = text.replace(')x', ')*x')
    text = text.replace(')(', ')*(')
    for repl in re.findall('\d+x', text):
        text = text.replace(repl, repl[:-1] + '*x')
    for repl in re.findall('\d+\(', text):
        text = text.replace(repl, repl[:-1] + '*(')
    text_list = text.split('=')
    text = text_list[0]+'-'+'('+text_list[1]+')'
    x = sympy.Symbol('x')
    try:
        result = str(sympy.solve(text, x)[0])
    except:
        raise EquationInvalidError('方程不合法，请检查')
    return result

def get_length(point1, point2): #kk:两点欧氏距离
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def italic_to_rec(img, east_box): # kk:透视变化将斜体框转化为矩形水平框
    h = img.shape[0]
    w = img.shape[1]
    for i in range(4):
        east_box[2 * i] = max(east_box[2 * i], 0)
        east_box[2 * i] = min(east_box[2 * i], w)
        east_box[2 * i + 1] = max(east_box[2 * i + 1], 0)
        east_box[2 * i + 1] = min(east_box[2 * i + 1], h)
    point0 = [east_box[0], east_box[1]]
    point1 = [east_box[2], east_box[3]]
    point2 = [east_box[4], east_box[5]]
    point3 = [east_box[6], east_box[7]]
    w_up = get_length(point0, point1)
    w_down = get_length(point2, point3)
    h_left = get_length(point3, point0)
    h_right = get_length(point1, point2)
    w = max(w_up, w_down)
    h = max(h_left, h_right)
    pts1 = np.float32([point0, point1, point2, point3])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (int(w), int(h)))
    return dst


def expansion_box(east_box, factor = 0.1):
    index = [4, 5, 6, 7, 0, 1, 2, 3]
    east_box = np.array(east_box)
    vector = east_box - east_box[index]
    east_box_new = factor * vector + east_box
    return east_box_new


if __name__ == "__main__":
    print(judge("60元-56元=4元", type = 'HS'))


