#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
from customized_answer_model.src.answer_model import customization_model
from general_answer_model.src.general_model import jieda_general_answer_solution
from procedure_model.src.procedure_common_model import procedure_solution


# NLP批改的入口主函数，包括批改过程和批改答案
def nlp_solution(question_id, ocr, step, _type, is_new_question, answer_online, procedure_online, answer_configuration):
    '''
    answer_online:线上答案
        type: list
        sample: ['19','21','33']
    procedure_online:线上关键步骤
        type: list
        sample: ['1+1=2',1+2=3']

    answer_configuration: 读取算法维护的数据库得到的字段,包含定制化所需的策略函数名、各种字段数据等
        type: dict
        sample:
            key: question_id
            value: {'solution_name': 'solution_A', 'correct_answer': ['19'], 'wrong_answer':[], 'keywords':['答', '可能是']}
    '''
    result = {}
    """
    {"msg":"success","code":20000,"data":{"text":[{"location":[92,842,225,893],"content":"说现了"},{"location":[106,549,381,592],"content":"比前项后项比号"},{"location":[99,602,388,645],"content":"除法被除号除数除号"},{"location":[85,496,431,539],"content":"⑤比除黛法分数的关系。"},{"location":[205,293,331,336],"content":"a:b=-b"},{"location":[92,443,473,487],"content":"②地值:比的值(算式的结果)"},{"location":[99,797,488,848],"content":"(最简整数比了前后项互质的比。"},{"location":[106,338,509,389],"content":"两千数相除也叫做两个数的比"},{"location":[98,646,538,703],"content":"分影分子分母分数线分数值。"},{"location":[92,752,743,795],"content":"⑤比的化简:前项后项直蘋约去公的因数的过程(的分)"},{"location":[397,556,459,592],"content":"比值"},{"location":[99,699,772,743],"content":"④比的性质前项后项同时除以(乘以)一个不为0的数冲值不变"}],"char_count":146},"requestId":"1578984795646666656081271132160"}
    """
    try:
        ocr = eval(ocr)
    except TypeError:
        ocr = ocr
    stu_ans = ''
    for line in ocr.get('text'):
        stu_ans += line.get('content')
        stu_ans += '\n'

    if step == 1:  # 需要判过程
        procedure_ret= procedure_solution(stu_ans, answer_online, procedure_online)
    else:  # 如果不需要批改过程，返回过程状态为
        procedure_ret = {
            'procedure_status': '100',
            'procedure_label': [],
            'procedure_list': []
        }

    if _type == 0 or _type == 1:  # 如果为选择填空
        answer_prediction, answer_line = jieda_general_answer_solution(stu_ans, answer_online)  # 选择填空批改答案通用模型
    else:  # 如果为解答题
        if not is_new_question:
            answer_prediction, answer_line = jieda_general_answer_solution(stu_ans, answer_online)  # 有改动的题目走解答题通用模型
        else:
            # 未改动的走解答题定制化模型
            answer_prediction, answer_line = customization_model(answer_configuration, question_id, stu_ans)

    result['answer_label'] = answer_prediction
    result['answer'] = answer_line
    result.update(procedure_ret)
    return result


def main():
    # 下面给一个例子，线上需要从算法维护的定制化数据库里一次性把数据读取并转化成以下结构
    answer_configuration = {}
    answer_configuration['1bcf539d03934d4fbcacc2187b49f937'] = {'solution_name': 'solution_A', 'correct_answer': ['19'],
                                                                'wrong_answer': [], 'keywords': ['答', '可能是']}
    answer_configuration['c35ea3c4964746f18f60154203ecedca'] = {'solution_name': 'solution_B', 'correct_answer': ['36'],
                                                                'wrong_answer': ['-36'], 'keywords': []}
    answer_configuration['c3cab6700ce847c689f53bf59161d482'] = {'solution_name': 'solution_C',
                                                                'correct_answer': ['x=2', 'y=3', 'x=11', 'y=1'],
                                                                'wrong_answer': [], 'keywords': []}
    answer_configuration['edcc1a6ab70048a1b062a2491ecdb47a'] = {'solution_name': 'solution_D',
                                                                'correct_answer': {'大桶': '20', '小桶': '30'},
                                                                'wrong_answer': [], 'keywords': ['答']}
    answer_configuration['8f663413cd874b64b3f3e68ba01e2082'] = {'solution_name': 'solution_E',
                                                                'correct_answer': ['5月16', '5.16'], 'wrong_answer': [],
                                                                'keywords': ['答']}
    answer_configuration['d8709f5b531847cfacc0d4853ec0b7dd'] = {'solution_name': 'solution_F',
                                                                'correct_answer': ['周三', '星期三', '星期3', '周3'],
                                                                'wrong_answer': [], 'keywords': ['答']}
    ocr = {"ans": [], "text": [{"location": [335, 51, 579, 100], "content": "⑥!与,", "timestamp": "0000000000"},
                               {"location": [63, 742, 307, 786], "content": "∴∠CDE=\\frac{1}{2}∠BDC=90°-\\frac{a}{2}",
                                "timestamp": "0000000000"},
                               {"location": [80, 973, 249, 1019], "content": "∴∠DE=90°+90-b}{2}",
                                "timestamp": "0000000000"},
                               {"location": [47, 904, 294, 950], "content": "():∠CD=90°,设∠ADO=6.",
                                "timestamp": "0000000000"},
                               {"location": [43, 576, 232, 620], "content": "1,原,", "timestamp": "0000000000"},
                               {"location": [75, 941, 265, 982], "content": "∴∠CDE=\\frac{1880-0)}{5}=\\frac{(a0-b}{2}",
                                "timestamp": "0000000000"},
                               {"location": [87, 1171, 298, 1214], "content": "又∵∠DDEF+\\frac{D}{2}=180°",
                                "timestamp": "0000000000"},
                               {"location": [64, 813, 181, 861], "content": "∴∠DOE=\\frac{a}{2}",
                                "timestamp": "0000000000"},
                               {"location": [60, 666, 250, 704], "content": "∴∠ADC+∠BOC=180°",
                                "timestamp": "0000000000"},
                               {"location": [44, 784, 175, 821], "content": "又∵∠CDD=90°", "timestamp": "0000000000"},
                               {"location": [64, 709, 217, 749], "content": "∴∠BDC=180°-a", "timestamp": "0000000000"},
                               {"location": [72, 1095, 215, 1134], "content": "=\\frac{20-)}{2}-45+\\frac{b}{2}",
                                "timestamp": "0000000000"},
                               {"location": [66, 868, 211, 902], "content": "∴∠ADC=∠EOD", "timestamp": "0000000000"},
                               {"location": [74, 1053, 209, 1096], "content": "∠DOE=185°-\\frac{6}{2}",
                                "timestamp": "0000000000"},
                               {"location": [40, 624, 189, 657], "content": "乙.(ADC=D", "timestamp": "0000000000"},
                               {"location": [277, 125, 638, 167], "content": "B,DM∠ADB,DN平分∠ADC,求MN,",
                                "timestamp": "0000000000"},
                               {"location": [84, 1017, 218, 1051], "content": "∴∠ADC=90°+6", "timestamp": "0000000000"},
                               {"location": [259, 480, 403, 589], "content": "B", "timestamp": "0000000000"},
                               {"location": [497, 377, 649, 435], "content": "∠MN=\\frac{a-3}{2}",
                                "timestamp": "0000000000"},
                               {"location": [55, 203, 203, 324], "content": "答∠", "timestamp": "0000000000"},
                               {"location": [483, 487, 634, 556], "content": "Mn=\\frac{8-7}{2}",
                                "timestamp": "0000000000"},
                               {"location": [54, 126, 194, 165], "content": "1)∠ADB=Q", "timestamp": "0000000000"},
                               {"location": [520, 227, 652, 292], "content": "∠MN=\\frac{a+3}{2}",
                                "timestamp": "0000000000"},
                               {"location": [196, 129, 307, 163], "content": "∠ADC=B", "timestamp": "0000000000"}]}
    # 调用NLP批改入口函数
    ret = nlp_solution('1bcf539d03934d4fbcacc2187b49f937', ocr, 1, 0, 0, ['19', '21', '33'], ['1+1=2', '1+2=3'],
                       answer_configuration)
    print(ret)


if __name__ == '__main__':
    main()
