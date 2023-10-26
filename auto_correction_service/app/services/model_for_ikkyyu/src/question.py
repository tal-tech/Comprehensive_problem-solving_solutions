from util import solve_equation, islegal, isdigit
from util import Result, judge, crossJudge
import re


class Question(object):
    def __init__(self, Bbox, type):
        self.indexInPaper = Bbox.indexInPaper
        self.state = Result.problem  # 判题结果
        self.output = ""  # 显示的题目结果
        self.recognition_result = ""  # 给测试传回识别结果
        self.type = type # 1 横式 2 有题干竖式 3 无题干竖式 4脱式
    def getPaperIndex(self):
        return self.indexInPaper


class HSQuestion(Question):
    def __init__(self, HSBbox):
        Question.__init__(self, HSBbox, 1)
        self.stem = HSBbox
        # self.outputText = None

    def revise(self):
        HStext = self.stem.greedySearchResult
        self.state, self.output = judge(HStext)
        self.recognition_result = HStext

    def doubleCheck(self):
        HStexts = self.stem.beamSearchResult
        for index, HStext in enumerate(HStexts):
            return_state, return_output = judge(HStext)
            if return_state == Result.correct:
                self.state = Result.correct
                self.output = return_output
                self.recognition_result = HStext
                return
            # elif return_state == Result.incorrect: #kk:doublecheck判对才修改状态,防止一些没意义的式子在doublecheck里被判为incorrect
            #     self.state = Result.incorrect


class SSQuestion(Question):
    def __init__(self, SSBbox, SSSolutionList, SSStem=None):
        self.stem = SSStem
        if self.stem:
            Question.__init__(self, SSBbox, 2)
        else:
            Question.__init__(self, SSBbox, 3)
        self.solutionList = SSSolutionList

    def revise(self):
        if self.stem:  # 有题干竖式判题
            if '=' in self.stem.greedySearchResult or '≈' in self.stem.greedySearchResult:
                self.state, self.output = judge(self.stem.greedySearchResult, type='SS')
                if self.state == Result.correct:
                    self.recognition_result = self.stem.greedySearchResult
                    return

            if '≈' in self.stem.greedySearchResult: #kk:题干带约等于
                symbol = '≈'
            else:
                symbol = '='
            former = ''
            stemList = []
            ansList = []
            yushuList = []
            if symbol in self.stem.greedySearchResult:
                former = self.stem.greedySearchResult.split(symbol)[0]
                ans_temp = self.stem.greedySearchResult.split(symbol)[-1]
                if '*' in ans_temp:
                    ans = ans_temp.split('*')[0]
                    yushu = ans_temp.split('*')[-1]
                elif '~' in ans_temp:
                    ans = ans_temp.split('~')[0]
                    yushu = ans_temp.split('~')[-1]
                else:
                    ans = ans_temp
                    yushu = ''

                if former and islegal(former, level=1):
                    stemList.append(former)
                if ans and isdigit(ans):
                    ansList.append(ans)
                if yushu and isdigit(yushu):
                    yushuList.append(yushu)
            else:
                if islegal(self.stem.greedySearchResult, level=1):
                    former = self.stem.greedySearchResult
                    stemList.append(former)

            if '÷' in self.stem.greedySearchResult:  #kk:除法竖式
                # kk：除法竖式从上往下搜答案
                if isdigit(self.solutionList[0].greedySearchResult):
                    ansList.append(self.solutionList[0].greedySearchResult)
                elif isdigit(self.solutionList[1].greedySearchResult) and (not '@' in self.solutionList[0].greedySearchResult):
                    ansList.append(self.solutionList[1].greedySearchResult)
                flag = True
                for index, bbox in enumerate(self.solutionList):  # kk:初步支持带题干除法竖式判题
                    if '@' in bbox.greedySearchResult:
                        beichushu = bbox.greedySearchResult.split('@')[-1]
                        chushu = bbox.greedySearchResult.split('@')[0]
                        stemList.append(beichushu + '÷' + chushu)
                        if stemList[0]:
                            stemList.append(beichushu + '÷' + stemList[0].split('÷')[1])
                            stemList.append(stemList[0].split('÷')[0] + '÷' + chushu)
                        break
                    if index == len(self.solutionList) - 1:
                        flag = False
                for yushu in self.solutionList[-1:-3:-1]:
                    if isdigit(yushu.greedySearchResult):
                        yushuList.append(yushu.greedySearchResult)
                    if not flag and yushu.greedySearchResult[0] == '+':
                        yushuList.append(yushu.greedySearchResult[1:])

                self.state, self.output = crossJudge(stemList, ansList, yushuList, type='SS')
                self.recognition_result = self.output

            else: #加减乘竖式
                index = 0
                while(not isdigit(self.solutionList[index].greedySearchResult)): #kk:排除第一行框到题干的情况
                   index +=1
                # while(self.solutionList[index].greedySearchResult == self.stem.greedySearchResult):
                #     index +=1
                firstNum = self.solutionList[index].greedySearchResult

                if firstNum[0] in '+-×': #kk:如果竖式第一行没被框出来，框出来的第一行带运算符
                    if stemList and (firstNum[0] in stemList[0]):
                        if former and isdigit(firstNum[1:]):
                            stemList.append(former.split(firstNum[0])[0] + firstNum)

                elif isdigit(firstNum):
                    operators = list()
                    for solutionCandidate in self.solutionList[index+1:]:
                        operator = re.search(r'[\+\-×]\d+\.?\d*', solutionCandidate.greedySearchResult)
                        if operator:
                            operators.append(operator.group())

                    if len(operators) >= 1:
                        secondNum = ''
                        for operator in operators:
                            secondNum += operator
                        op = secondNum[0]
                        if op in '+-×':
                            stemList.append(firstNum + secondNum)
                            if former and (op in self.stem.greedySearchResult):
                                stemList.append(firstNum + op + former.split(op)[1])
                                stemList.append(former.split(op)[0] + op + secondNum[1:])
                                if op != '-' and not (set('+-×') & set(secondNum[1:])): # kk:乘法加法把两个运算数交换一下进入题干序列
                                    stemList.append(secondNum[1:] + op + former.split(op)[1])
                                    stemList.append(firstNum + op + former.split(op)[0])
                    else: # kk:答案没框全
                        for op in ['+', '-', '×']:
                            if former and (op in former):
                                stemList.append(firstNum + op + former.split(op)[1])
                                if op != '-':
                                    stemList.append(former.split(op)[0] + op + firstNum)

                solutionLen = len(self.solutionList)  # kk:有可能不够3行
                end = min(solutionLen, 2)

                for solutionCandidate in self.solutionList[-1:-(end+1):-1]:  # 非带余竖式查后三个
                    # solutionCandidateResult = re.search(r'\d+\.?\d*', solutionCandidate.greedySearchResult)
                    # constructed = former + '=' + solutionCandidate.greedySearchResult
                    # return_state, return_output = judge(constructed, type='SS')
                    # if return_state == Result.correct:
                    #     self.state = Result.correct
                    #     self.output = return_output
                    #     self.recognition_result = constructed
                    #     return
                    if isdigit(solutionCandidate.greedySearchResult):
                        ansList.append(solutionCandidate.greedySearchResult)
                self.state, self.output = crossJudge(stemList, ansList, type='SS')
                self.recognition_result = self.output

        else:  # 无题干竖式判题(目前仅支持两元计算，多元运算todo)
            if len(self.solutionList) < 3:  # kk:竖式解答过程少于2行,不判
                self.state = Result.problem
                return

            for index, bbox in enumerate(self.solutionList): # kk:初步支持无题干除法竖式判题
                if '@' in bbox.greedySearchResult:
                    if index == 0:
                        self.state = Result.problem
                        return
                    beichushu = bbox.greedySearchResult.split('@')[-1]
                    chushu = bbox.greedySearchResult.split('@')[0]
                    constructed = beichushu + '÷' + chushu + '=' + shang # kk:shang为上一个框的识别结果
                    return_state, return_output = judge(constructed, type='SS')
                    self.state = return_state
                    self.output = return_output
                    self.recognition_result = constructed
                    if return_state != Result.correct:
                        solutionLen = len(self.solutionList)  # kk:有可能不够3行
                        end = min(solutionLen, 2)
                        for yushu in self.solutionList[-1:-(end + 1):-1]:
                            if isdigit(yushu.greedySearchResult):
                                equation = constructed + '***' + yushu.greedySearchResult
                                return_state, return_output = judge(equation, type='SS')
                                if return_state == Result.correct:
                                    self.state = Result.correct
                                    self.output = return_output
                                    self.recognition_result = equation
                                    return
                    return
                shang = bbox.greedySearchResult
            # kk:不是除法竖式
            operators = list()
            for bbox in self.solutionList:
                text = bbox.greedySearchResult
                operator = re.search(r'[\+\-×]\d+\.?\d*', text) #kk:只找有运算符的行
                # operator = re.fullmatch(r'[\+-×]?\d+\.?\d*', text) #kk:可能是因为末尾带空格,无法全匹配
                if operator:
                    operators.append(operator.group())
                # if len(operators) == 2: # todo: 多元运算
                #     break
            if len(operators) < 1:
                return
            former = self.solutionList[0].greedySearchResult
            if former[0] in '+-×': # kk：没框出第一行
                return
            for operator in operators:
                former += operator

            stemList = []
            ansList = []
            stemList.append(former)

            solutionLen = len(self.solutionList)  # kk:有可能不够3行
            end = min(solutionLen, 2)
            for solutionCandidate in self.solutionList[-1:-(end + 1):-1]:
                if isdigit(solutionCandidate.greedySearchResult):
                    ansList.append(solutionCandidate.greedySearchResult)

            self.state, self.output = crossJudge(stemList, ansList, type='SS')
            self.recognition_result = self.output

    def doubleCheck(self):
        if self.stem:  # 有题干竖式
            for result in self.stem.beamSearchResult:
                if '=' in result:
                    return_state, return_output = judge(result)  # 非带余竖式查后最后一个
                    if return_state == Result.correct:
                        self.state = Result.correct
                        self.output = return_output
                        self.recognition_result = result
                        return
            for former in self.stem.beamSearchResult:
                former = former.split('=')[0]
                if '÷' in former:  # 带余竖式
                    for quotient in self.solutionList[0].beamSearchResult:
                        if isdigit(quotient):
                            for residue in self.solutionList[-1].beamSearchResult:
                                if isdigit(residue):
                                    return_state, return_output = judge(former + '=' + quotient + '***' + residue)
                                    if return_state == Result.correct:
                                        self.state = Result.correct
                                        self.output = return_output
                                        self.recognition_result = former + '=' + quotient + '***' + residue
                                        return
                else:
                    for result in self.solutionList[-1].beamSearchResult:
                        if isdigit(result):
                            return_state, return_output = judge(former + '=' + result)  # 非带余竖式查后最后一个
                            if return_state == Result.correct:
                                self.state = Result.correct
                                self.output = return_output
                                self.recognition_result = former + '=' + result
                                return
        else:  # 无题干竖式
            if len(self.solutionList) < 3:  # kk:竖式解答过程少于3行,不判
                self.state = Result.problem
                return
            stemList = []
            beichushuList = []
            chushuList = []
            ansList = []
            #--------------------------除法竖式doublecheck begin-----------------------
            for result in self.solutionList[1].beamSearchResult:
                if '@' in result:
                    beichushu = result.split('@')[-1]
                    chushu = result.split('@')[0]
                    if isdigit(beichushu):
                        beichushuList.append(beichushu)
                    if isdigit(chushu):
                        chushuList.append(chushu)
            if beichushuList and chushuList:
                yushuList = []
                for shang in self.solutionList[0].beamSearchResult:
                    if isdigit(shang):
                        ansList.append(shang)
                for yushu in self.solutionList[-1].beamSearchResult:
                    if isdigit(yushu):
                        yushuList.append(yushu)
                for beichushu in beichushuList:
                    for chushu in chushuList:
                        stemList.append(beichushu + '÷' + chushu)
                state, output = crossJudge(stemList, ansList, yushuList, type='SS')
                if state == Result.correct:
                    self.state = state
                    self.output = output
                    self.recognition_result = self.output
                return
            # ------------------------除法竖式doublecheck  end-----------------------

            for operator1 in self.solutionList[0].beamSearchResult:
                operator1 = re.fullmatch(r'[\+\-×]?\d+\.?\d*', operator1.strip()) # todo: 修改竖式判题策略
                if not operator1:
                    continue
                operator1 = operator1.group()
                for operator2 in self.solutionList[1].beamSearchResult:
                    operator2 = re.fullmatch(r'[\+\-×]?\d+\.?\d*', operator2.strip())
                    if not operator2:
                        continue
                    operator2 = operator2.group()
                    former = operator1 + operator2
                    for solution in self.solutionList[-1].beamSearchResult:
                        if isdigit(solution):
                            equation = former + '=' + solution
                            return_state, return_output = judge(equation)
                            if return_state == Result.correct:
                                self.state = Result.correct
                                self.output = return_output
                                self.recognition_result = equation
                                return


class TSQuestion(Question):
    def __init__(self, TSBbox, TSSolutionList, TSStem):
        Question.__init__(self, TSBbox, 4)
        self.stem = TSStem
        self.solutionList = TSSolutionList

    def revise(self):
        if re.search(r'[a-zA-Z]', self.stem.greedySearchResult) and '=' in self.stem.greedySearchResult:  # kk:解方程需要有x和=
            # if set('{}') & set(self.stem.greedySearchResult):  # kk:分数先不管
            #     self.state = Result.problem
            #     return
            if len(self.solutionList) > 0 and self.solutionList[0].greedySearchResult[0] != '=':  # kk:滤掉带字母的脱式
                character = re.search(r'[a-zA-Z]', self.stem.greedySearchResult).group()
                self.stem.greedySearchResult = self.stem.greedySearchResult.replace(character, 'x')
                try:
                    solution = solve_equation(self.stem.greedySearchResult)
                    solutionLen = len(self.solutionList)
                    end = min(solutionLen, 3)
                    for index, solutionCandidate in enumerate(self.solutionList[-1:-(end + 1):-1]):
                        result = re.search(r'-?\d+\.?\d*$', solutionCandidate.greedySearchResult)
                        # result = solutionCandidate.greedySearchResult.split('=')[-1]
                        if result:
                            result = result.group()
                            return_state, return_output = judge(solution + '=' + result, type='TS')
                            # return_state, return_output = judge(result.group() + '=' + solution, type = 'TS')
                            if return_state == Result.correct:
                                self.state = Result.correct
                                self.output = self.stem.greedySearchResult + ' ' + '解为x=' + result  # .group()
                                self.recognition_result = self.output
                                # HSList = []
                                # for HSRest in self.solutionList[-1:index]:
                                #     HSList.append(HSRest)
                                return  # HSList
                            # if result == solutionCandidate.greedySearchResult.split('=')[-1]:
                            #     #如果最后一行是一个数,但答案不对,直接判错,不再往上搜索
                            #     self.output = self.stem.greedySearchResult + ' ' + '解为x=' + '%f' % (float(result))
                            #     self.recognition_result = self.output
                            #     return
                    self.state = Result.incorrect
                    self.output = self.stem.greedySearchResult + ' ' + '解为x=' + self.solutionList[-1].greedySearchResult.split('=')[-1]
                    self.recognition_result = self.output
                except Exception as e:
                    pass
            # else: #kk: 题目的默认状态为problem
            #     self.state = Result.problem
        else:
            index = 0
            #kk:如果第一行不是题干,往下搜直到找到题干 07.04
            while not (set(self.stem.greedySearchResult) & set('+-×÷')):
                self.stem.greedySearchResult = self.solutionList[index].greedySearchResult
                index += 1

            if '=' in self.stem.greedySearchResult:
                return_state, return_output = judge(self.stem.greedySearchResult)
                # 先直接判题干,如果题干有答案且正确,直接判对
                if return_state == Result.correct:
                    self.state = Result.correct
                    self.output = return_output
                    self.recognition_result = self.output
                    return

            if self.stem.greedySearchResult[0] == '=':
                self.stem.greedySearchResult = self.stem.greedySearchResult[1:]

            former = self.stem.greedySearchResult.split('=')[0]
            if former == '':
                self.state = Result.problem
                return
            solutionLen = len(self.solutionList)
            end = min(solutionLen, 3)
            digit = ''
            for index, solutionCandidate in enumerate(self.solutionList[-1:-(end + 1):-1]):
                if not solutionCandidate.greedySearchResult:
                    continue
                symbol = '='
                if '=' in solutionCandidate.greedySearchResult:
                    result = solutionCandidate.greedySearchResult.split('=')[-1]
                elif '≈' in solutionCandidate.greedySearchResult:
                    result = solutionCandidate.greedySearchResult.split('≈')[-1]
                    symbol = '≈'
                elif isdigit(solutionCandidate.greedySearchResult):
                    result = solutionCandidate.greedySearchResult
                else:
                    continue
                # result = re.search(r'-?\d+\.?\d*', solutionCandidate.greedySearchResult)
                # if result:
                #     result = result.group()
                if digit == "":
                    if result == solutionCandidate.greedySearchResult:
                        digit = result
                    if len(solutionCandidate.greedySearchResult) > 1:
                        if result == solutionCandidate.greedySearchResult[1:]:
                            digit = result

                if solutionCandidate.greedySearchResult == result and result[0] == '-': #kk:脱式最后一行把等号识别成了负号
                    return_state, return_output = judge(former + symbol + result[1:], type='TS')
                    if return_state == Result.correct:
                        self.state = Result.correct
                        self.output = return_output
                        self.recognition_result = former + symbol + result[1:]
                        return

                return_state, return_output = judge(former + symbol + result, type='TS')
                if return_state == Result.correct:
                    self.state = Result.correct
                    self.output = return_output
                    self.recognition_result = former + symbol + solutionCandidate.greedySearchResult.split(symbol)[-1]
                    return
                # if not set('+-*/') & set(solutionCandidate.greedySearchResult.split('=')[-1]):
                #     # kk:如果最后一行是一个数,但答案不对,直接判错,不再往上搜索
                #     self.output = former + '=' + self.solutionList[-1].greedySearchResult.split('=')[-1]
                #     self.recognition_result = self.output
                #     return

            if self.solutionList[-1].greedySearchResult[0] == '≈':  # kk:脱式约等于
                symbol = '≈'
            else:
                symbol = '='
            self.state = Result.incorrect
            self.output = former + symbol + digit
            self.recognition_result = self.output

    def doubleCheck(self):
        for stem in self.stem.beamSearchResult:
            if 'x' in stem:  # 解方程
                try:
                    solution = solve_equation(stem)
                except Exception as e:
                    continue
                for solutionCandidate in self.solutionList[-1].beamSearchResult:
                    # result = re.search(r'[+-]?\d+\.?\d*$', solutionCandidate) #kk:直接split"=",不用search
                    result = solutionCandidate.split('=')[-1]
                    if result:
                        return_state, return_output = judge(result + '=' + solution)
                        # return_state, return_output = judge(result.group + '=' + solution)
                        if return_state == Result.correct:
                            self.state = Result.correct
                            self.output = stem + ' ' + '解为x=' + result
                            self.recognition_result = stem + ' ' + '解为x=' + result
                            return
            else:
                if '=' in stem:
                    return_state, return_output = judge(stem)
                    # 先直接判题干,如果题干有答案且 正确,直接判对
                    if return_state == Result.correct:
                        self.state = Result.correct
                        self.output = return_output
                        self.recognition_result = self.output
                        return
                former = stem.split('=')[0]
                if former == '':
                    continue
                for solutionCandidate in self.solutionList[-1].beamSearchResult:
                    return_state, return_output = judge(former + '=' + solutionCandidate.split('=')[-1])
                    if return_state == Result.correct:
                        self.state = Result.correct
                        self.output = return_output
                        self.recognition_result = former + '=' + solutionCandidate.split('=')[-1]
                        return
