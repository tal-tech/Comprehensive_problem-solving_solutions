import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__)+'/../')
from ssd_for_ikkyyu.inference_with_east import det_Layout, get_det_result, get_post_result, visulization
from east_for_ikkyyu.east_inference import East
from crnn_for_ikkyyu.model import CTC_Model
from attention_for_ikkyyu.interface import Dense_net
from paper import Paper
from question import HSQuestion, SSQuestion, TSQuestion
from visualization import visualization
from util import Result, islegal, fraction_cvt
import cv2
import time
import numpy as np
import logging
from yaml_cfg import cfg
from app.common.tool_unit import func_time


class pipeline(object):
    def __init__(self, gpu_id='0'):

        '''logger initialization--------------------------------------------------------------'''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("pipeline_log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        '''----------------------------------------------------------------------------------'''

        self.logger.info("start initial pipeline")
        self.gpu_id = gpu_id
        self.attention = False
        self.expansion = False
        # update by boby 20190515
        # 采用读取yaml文件的方式进行模型路径的配置
        dir_path = os.path.dirname(__file__)+'/../'
        detection_model_path = cfg["dependences"][4]["model_path"]
        recognition_model_path = cfg["dependences"][5]["model_path"]
        east_model_path = cfg["dependences"][6]["model_path"]
        attention_model_path = cfg["dependences"][7]["model_path"]
        torch_slim_path = cfg["dependences"][7]["torch_slim_path"]
        detection_model_path = dir_path + detection_model_path
        recognition_model_path = dir_path + recognition_model_path
        east_model_path = dir_path + east_model_path
        attention_model_path = dir_path + attention_model_path
        torch_slim_path = dir_path + torch_slim_path

        self.detector = det_Layout(deviceid = gpu_id, CKPT_PATH = detection_model_path)  # 初始化检测模型
        self.recognition_model = CTC_Model(model_path = recognition_model_path)  # 初始化识别模型
        if self.attention:
            self.attention_model = Dense_net(token_path=torch_slim_path, checkpoint_path=attention_model_path,
                                             gpu_index=0)
        else:
            self.attention = None
        self.east = East(CKPT_PATH=east_model_path)  # 初始化检测模型
        self.ret = dict()  # analyse执行后返回给grpc的结果
        self.visual = False  # 结果是否可视化
        self.clock = True  # 是否输出各模块运行时间
        self.write = False  # 是否写各模块输出结果到文件
        self.aiqa = False  # 是否支持AIQA定制化输出，仅用于非线上环境开启
        self.aiqaAllTextLine = list()  # 记录 检测出的所有文本框 和 文本框识别内容

        # initial_image = np.zeros((512, 512, 3))  # 构造一张为零的图片
        # process(self.detector, initial_image, initial_image.shape)
        self.logger.info("initial pipeline complete")

    def analyse(self, image, imgname, recognition_batch_size = 64):
        """
        grpc 函数调用-- 处理整个流程：检测--版面（paper）--识别--
        :param image:
        :param recognition_batch_size:
        :return:
        """
        visualizer_write_filename_generator = time.time()
          # 初始化版面
        visualizer = visualization(image)  # 初始化可视化

        # 开始 检测流程
        detection_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        paper = Paper(detection_image)
        if self.clock:
            start = time.time()

        # 返回结果：1.横式 2.竖式 3.脱式（含方程）
        # ----------------------------------kk:不用east的结果直接inference----------------
        # detectionResult = process(self.detector, detection_image, detection_image.shape)  # 调用检测接口
        classes, scores, bboxes = self.detector.inference(detection_image)
        ssd_rs = get_det_result(classes, scores, bboxes)
        east_rs = self.east.east_process(detection_image)
        detectionResult = get_post_result(ssd_rs, east_rs, detection_image.shape)

        if self.clock:
            end = time.time()
            print("检测用时{}s".format(end - start))
            self.logger.info("dect time {}s".format(end - start))
        # 结束 检测流程

        if self.clock:
            start = time.time()
        paper.loadBboxesfromjson(detectionResult, 0.01)  # 加载检测结果，初始化所有Bbox
        if self.clock:
            end = time.time()
            print("生成识别序列{}s".format(end - start))


#         # 开始 识别流程
        if self.clock:
            start = time.time()
        images, widths = paper.createRecognitionInput()  # 生成识别模型输入



        # output_list = []

        detection_count = 1  # 检测多少张图片，每次只能检测一张
        recognition_count = 0  # 识别都是小图，送进去多少张就识别多少张
        # print(len(images))
        # 修改成beamsearch
        # if len(images) != 0:
        #     for iteration in range((len(images) + recognition_batch_size - 1) // recognition_batch_size):
        #         # 保证显存，分批送入
        #         sub_output_list, conf_vec = self.recognition_model.beam_search_interface(
        #             images[iteration * recognition_batch_size:(1 + iteration) * recognition_batch_size, ...],
        #             widths[iteration * recognition_batch_size:(1 + iteration) * recognition_batch_size])
        #
        #         output_list.extend(sub_output_list)
        #
        #     output_list = np.array(output_list)
        #     output_list = list(map(lambda x: x[0], output_list))  # 因为是一个list，只需要取第一个


        output_list = []
        logits = []
        sequence_length = []
        if len(images) != 0:
            for iteration in range((len(images) + recognition_batch_size - 1) // recognition_batch_size):
                sub_output_list, sub_logits, sub_sequence_length = self.recognition_model.greedy_search(
                    images[iteration * recognition_batch_size:(1 + iteration) * recognition_batch_size, ...],
                    widths[iteration * recognition_batch_size:(1 + iteration) * recognition_batch_size])
                output_list.extend(sub_output_list)
                logits.extend(sub_logits)
                sequence_length.extend(sub_sequence_length)
            output_list = np.array(output_list)
            logits = np.array(logits)
            sequence_length = np.array(sequence_length)

            output_list = fraction_cvt(output_list)
            paper.fillInGreedySearchResult(output_list)  # 将识别的结果填回所有Bbox

            recognition_count = len(images)  # 输出json标记，识别都是小图，送进去多少张就识别多少张

        else:  # 图像为空
            self.paper = paper
            return

        if self.clock:
            end = time.time()
            print("识别用时{}s".format(end - start))
            self.logger.info("recog time {}s".format(end - start))
        # 结束  识别流程

        call_times = [detection_count, recognition_count]
        call_count = dict(zip(["detection", "recognition"], call_times))
        self.ret["call_count"] = call_count

        '''
        ----------------------------------------------------------------------------------------------------------
        仅限 AIQA 测试使用
        '''
        if self.aiqa:
            for idx, bbox in paper.recognitionIndexToBbox.items():
                self.aiqaAllTextLine.append({'id': idx, 'box': [bbox.left, bbox.top, bbox.right, bbox.bottom], 'text': bbox.greedySearchResult})

        '''
        ----------------------------------------------------------------------------------------------------------
        识别结果可视化
        '''
        resize_factor = 0.2
        if self.visual:
            for recognitionIndex, bbox in paper.recognitionIndexToBbox.items():
                visualizer.drawBbox(bbox)
                visualizer.text(bbox, bbox.greedySearchResult)
            for SSIndex in paper.waitForStem:  # 竖式无题干
                visualizer.drawBbox(paper.paperIndexToBbox[SSIndex], color=(0, 0, 255))
            for TSIndex in paper.alreadyHaveStem:  # 脱式
                visualizer.drawBbox(paper.paperIndexToBbox[TSIndex], color=(255, 0, 255))
            if self.write:
                visualizer.save('{}_recognition.jpg'.format(imgname))
                # visualizer.save('{}_recognition.jpg'.format(visualizer_write_filename_generator))
                # visualizer.save('{}_origin.jpg'.format(visualizer_write_filename_generator))
            else:
                visualizer.show('recognition', resize_factor)
            visualizer.reset()

        '''
        -----------------------------------------------------------------------------------------------------------
        '''

        if self.clock:
            start = time.time()
        paper.findStemForSS(distanceThreshold=0.05, similarityThreshold=0.23)  # 版面分析
        '''
        版面分析结果可视化
        '''
        if self.clock:
            end = time.time()
            print("策略版面用时{}s".format(end - start))
            self.logger.info("layout time {}s".format(end - start))

        if self.visual:
            for HSIndex in paper.allHS:  # 横式
                visualizer.drawBbox(paper.paperIndexToBbox[HSIndex], color=(255, 0, 0))
            for SSIndex in paper.SSWithStem:  # 竖式
                visualizer.drawBbox(paper.paperIndexToBbox[SSIndex], color=(0, 255, 0))
                visualizer.drawBbox(paper.paperIndexToBbox[paper.parentToChildren[SSIndex][-1]], color=(255, 255, 0))  # 题干
            for SSIndex in paper.waitForStem:  # 竖式无题干
                visualizer.drawBbox(paper.paperIndexToBbox[SSIndex], color=(0, 0, 255))
            for TSIndex in paper.alreadyHaveStem:  # 脱式
                visualizer.drawBbox(paper.paperIndexToBbox[TSIndex], color=(255, 0, 255))
                for TSchildIndex in paper.parentToChildren[TSIndex]:
                    visualizer.drawBbox(paper.paperIndexToBbox[TSchildIndex], color=(255, 255, 0))
            if self.write:
                # visualizer.save('{}_paper.jpg'.format(visualizer_write_filename_generator))
                visualizer.save('{}_paper.jpg'.format(imgname))
            else:
                visualizer.show('paper', resize_factor)
            visualizer.reset()
            # visualizer.save('paperResult.jpg')

        '''
        -----------------------------------------------------------------------------------------------------------
        '''
        '''
        判题
        '''
        if self.clock:
            start = time.time()
        doubleCheckQueue = list()  # 需要二次判题的题号
        beamSearchQueue = list()  # 需要beam search的小框

        for SSIndex in paper.SSWithStem:
            try:
                children = [paper.paperIndexToBbox[index] for index in paper.parentToChildren[SSIndex]]
                question = SSQuestion(paper.paperIndexToBbox[SSIndex], children[:-1], children[-1])
                paper.paperIndexToQuestion[SSIndex] = question
                question.revise()
                if question.state != Result.correct:
                    doubleCheckQueue.append(question.indexInPaper)
                    beamSearchQueue.append(question.stem.indexInPaper)
                    beamSearchQueue.append(question.solutionList[0].indexInPaper)
                    beamSearchQueue.append(question.solutionList[1].indexInPaper)
                    beamSearchQueue.append(question.solutionList[-1].indexInPaper)
            except Exception as e:
                continue

        # 不带题干竖式判题
        for SSIndex in paper.waitForStem:
            '''检测可能误检判断第一个框是否为竖式的题干(检测准确后可以删除)-----------------------------------------------'''
            try:
                children = [paper.paperIndexToBbox[index] for index in paper.parentToChildren[SSIndex]]
                if '=' in children[0].greedySearchResult:
                    question = SSQuestion(paper.paperIndexToBbox[SSIndex], children[1:], children[0])
                else:
                    question = SSQuestion(paper.paperIndexToBbox[SSIndex], children)
                paper.paperIndexToQuestion[SSIndex] = question
                question.revise()
                if question.state != Result.correct:
                    doubleCheckQueue.append(question.indexInPaper)
                    beamSearchQueue.append(question.solutionList[0].indexInPaper)
                    beamSearchQueue.append(question.solutionList[1].indexInPaper)
                    beamSearchQueue.append(question.solutionList[-1].indexInPaper)
            except Exception as e:
                continue

        # 脱式判题
        for TSIndex in paper.alreadyHaveStem:
            try:
                children = [paper.paperIndexToBbox[index] for index in paper.parentToChildren[TSIndex]]
                stem = children[0]
                # kk:有可能脱式多框了一排或者完全框错,框内全是横式
                hs_num = 0
                for box in children[-1:0:-1]:
                    if (islegal(box.greedySearchResult,3) or islegal(box.greedySearchResult,2)) and (not box.greedySearchResult[0] == '='):
                        paper.allHS.add(box.indexInPaper)
                        hs_num += 1
                    else:
                        break
                if hs_num == len(children) - 1:
                    if islegal(stem.greedySearchResult, 2) or islegal(box.greedySearchResult,2):
                        paper.allHS.add(stem.indexInPaper)
                    continue
                # kk:脱式如果有"疑似题号",先去除"疑似题号"判一遍,如果不对,或者没有"疑似题号",正常判,
                # 采用这种顺序是为了让字符输出为完整的识别结果
                if (stem.greedySearchResult[1] == '.' or stem.greedySearchResult[2] == '.') \
                        and len(stem.greedySearchResult) > 3:
                    QuestionNum = stem.greedySearchResult[: stem.greedySearchResult.find('.')+1]  #kk:摘离题号
                    stem.greedySearchResult = stem.greedySearchResult[stem.greedySearchResult.find('.') + 1:]
                    question = TSQuestion(paper.paperIndexToBbox[TSIndex], children[1:], stem)
                    paper.paperIndexToQuestion[TSIndex] = question
                    question.revise()

                    if question.state != Result.correct:
                        stem.greedySearchResult = QuestionNum + stem.greedySearchResult
                        question = TSQuestion(paper.paperIndexToBbox[TSIndex], children[1:], stem)
                        paper.paperIndexToQuestion[TSIndex] = question
                        question.revise()
                else:
                    question = TSQuestion(paper.paperIndexToBbox[TSIndex], children[1:], stem)
                    paper.paperIndexToQuestion[TSIndex] = question
                    question.revise()
                #kk: 题干或者答案里有分数就doublecheck
                if question.state != Result.correct:# and ('|' in stem.greedySearchResult or '|' in children[-1].greedySearchResult):
                    doubleCheckQueue.append(question.indexInPaper)
                    beamSearchQueue.append(question.stem.indexInPaper)
                    beamSearchQueue.append(question.solutionList[-1].indexInPaper)
            except Exception as e:
                continue

        expansionQueue = list()
        tripleCheckQueue = list()
            # 横式判题
        for HSIndex in paper.allHS:
            try:
                question = HSQuestion(paper.paperIndexToBbox[HSIndex])
                paper.paperIndexToQuestion[HSIndex] = question
                question.revise()
                # 如果题目不正确，进入二次检测，添加到beam search检查队列
                # kk: 一次判题不正确, 扩图再判一次
                if question.state != Result.correct:# and '|' in paper.paperIndexToBbox[HSIndex].greedySearchResult:

                    if self.expansion:
                        expIndex = paper.paperIndexToRecognitionIndex[HSIndex]
                        expansion_img = np.array(paper.recognitionExImgList[expIndex])
                        width = np.array([expansion_img.shape[1]])
                        expansion_img = expansion_img[np.newaxis,..., np.newaxis]
                        output, _, _ = self.recognition_model.greedy_search(expansion_img, width)
                        output = fraction_cvt(np.array(output))[0]
                        greedySearchResult = paper.paperIndexToBbox[HSIndex].greedySearchResult
                        state_temp = question.state
                        output_temp = question.output
                        recognition_result_temp = question.recognition_result
                        # kk:如果扩后的识别结果跟以前不一样就进判题策略
                        if output != greedySearchResult:
                            paper.paperIndexToBbox[HSIndex].greedySearchResult = output
                            question.revise()
                            # kk:如果题目还是被判为错,并且一开始的题目state不为problem,就把改题的识别结果和判题state恢复
                            if question.state != Result.correct:
                                paper.paperIndexToBbox[HSIndex].greedySearchResult = greedySearchResult
                                if state_temp != Result.problem:
                                    question.state = state_temp
                                    question.output = output_temp
                                    question.recognition_result = recognition_result_temp

                            else:
                                expansionQueue.append(HSIndex)
                        # else:
                        #     if self.attention:
                        #         img_index = paper.paperIndexToRecognitionIndex[HSIndex]
                        #         img = paper.recognitionImgList[img_index]
                        #         attentionResult = self.attention_model.recognize(img)[0]
                        #         if attentionResult != paper.paperIndexToBbox[HSIndex].greedySearchResult:
                        #             doubleCheckQueue.append(question.indexInPaper)
                        #             beamSearchQueue.append(question.stem.indexInPaper)
                        #         else:
                        #             tripleCheckQueue.append(HSIndex)
                        #     else:
                        #         doubleCheckQueue.append(question.indexInPaper)
                        #         beamSearchQueue.append(question.stem.indexInPaper)
                if question.state != Result.correct:
                    if self.attention:
                        img_index = paper.paperIndexToRecognitionIndex[HSIndex]
                        img = paper.recognitionImgList[img_index]
                        attentionResult = self.attention_model.recognize(img)[0]
                        if attentionResult != paper.paperIndexToBbox[HSIndex].greedySearchResult:
                            doubleCheckQueue.append(question.indexInPaper)
                            beamSearchQueue.append(question.stem.indexInPaper)
                        else:
                            tripleCheckQueue.append(HSIndex)
                    else:
                        doubleCheckQueue.append(question.indexInPaper)
                        beamSearchQueue.append(question.stem.indexInPaper)
            except Exception as e:
                continue

        if self.clock:
            end = time.time()
            print("一次判题用时{}s".format(end - start))
            self.logger.info("judging once {}s".format(end - start))

        # beam search批处理并回填
        if self.clock:
            start = time.time()

        beamSearchRecognitionIndex = list(map(lambda x: paper.paperIndexToRecognitionIndex[x], beamSearchQueue))
        logits_beam_search = logits[beamSearchRecognitionIndex, ...]
        sequence_length_beam_search = sequence_length[beamSearchRecognitionIndex]
        output_list, conf_vec = self.recognition_model.beam_search(logits_beam_search, sequence_length_beam_search, beamsearch_width = 1)

        for li in range(len(output_list)):
            for i in range(len(output_list[li])):
                    output_list[li][i] = output_list[li][i].replace('{(', '{')
                    output_list[li][i] = output_list[li][i].replace(')}', '}')
                    output_list[li][i] = output_list[li][i].replace(')(', '|')
                    output_list[li][i] = output_list[li][i].replace(' ', '')

        paper.fillInBeamSearchResult(beamSearchQueue, output_list, conf_vec)

        if self.clock:
            end = time.time()
            print("beam search用时{}s".format(end - start))
            self.logger.info("beam search time {}s".format(end - start))


        # tripleCheckQueue = list()
        # attentionQueue = list()  # 需要attention识别的小框
        # # 再判题
        if self.clock:
            start = time.time()
        for questionIndex in doubleCheckQueue:
            try:
                paper.paperIndexToQuestion[questionIndex].doubleCheck()
                # question = paper.paperIndexToQuestion[questionIndex]
                # if question.state != Result.correct:
                #     tripleCheckQueue.append(questionIndex)
                #     if question.type == 1:
                #         attentionQueue.append(question.stem.indexInPaper)
                #     else:
                #         if question.type != 3:
                #             attentionQueue.append(question.stem.indexInPaper)
                #         for box in question.solutionList:
                #             attentionQueue.append(box.indexInPaper)
            except Exception as e:
                continue
        if self.clock:
            end = time.time()
            print("二次判题用时{}s".format(end - start))
            self.logger.info("second judging time {}s".format(end - start))



        # 190711 kk: attention识别三次判题序列框得到结果替换greedysearch的结果
        # if self.clock:
        #     start = time.time()
        # for boxIndex in attentionQueue:
        #     try:
        #         img_index = paper.paperIndexToRecognitionIndex[boxIndex]
        #         img = paper.recognitionImgList[img_index]
        #         a = paper.paperIndexToBbox[boxIndex].greedySearchResult
        #         paper.paperIndexToBbox[boxIndex].greedySearchResult = self.attention_model.recognize(img)[0]
        #     except Exception as e:
        #         continue
        # if self.clock:
        #     end = time.time()
        #     print("attention识别用时{}s".format(end - start))
        #     self.logger.info("third judging time {}s".format(end - start))
        #
        #
        # # 190711 kk: attention识别结果的判题, 直接使用一次判题的接口
        # if self.clock:
        #     start = time.time()
        # for questionIndex in tripleCheckQueue:
        #     try:
        #         question = paper.paperIndexToQuestion[questionIndex]
        #         #190711 kk: 只有attention模型判对了才采用该结果
        #         # state = question.state
        #         # output = question.output
        #         question.revise()
        #         if question.state == Result.correct:
        #             print(imgname)
        #         # if question.state != Result.correct:
        #         #     # print('wrong:', question.output)
        #         #     question.state = state
        #         #     question.output = output
        #         #     question.recognition_result = output
        #         # else:
        #             # print('right:', output)
        #     except Exception as e:
        #         continue
        # if self.clock:
        #     end = time.time()
        #     print("三次判题用时{}s".format(end - start))
        #     self.logger.info("third judging time {}s".format(end - start))





        '''------------------------------------------------------------------------------------------
        判题结果可视化
        --------------------------------------------------------------------------------------------'''
        if self.visual:
            for question in paper.paperIndexToQuestion.values():
                if question.state == Result.correct:
                    if question.indexInPaper in tripleCheckQueue:  #一次判对绿框,二次判对蓝框
                        color = (255, 0, 0)
                    elif question.indexInPaper in expansionQueue:
                        color = (255, 255, 255)
                    elif question.indexInPaper in doubleCheckQueue:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 0)
                    visualizer.drawBbox(paper.paperIndexToBbox[question.indexInPaper], color = color)
                    visualizer.text(paper.paperIndexToBbox[question.indexInPaper], question.output)
                elif question.state == Result.incorrect:
                    if question.indexInPaper in tripleCheckQueue:  #一次判错红框,二次判对黄框
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    visualizer.drawBbox(paper.paperIndexToBbox[question.indexInPaper], color = color)
                    visualizer.text(paper.paperIndexToBbox[question.indexInPaper], question.output)
            if self.write:
                # visualizer.save('{}_judgement.jpg'.format(visualizer_write_filename_generator))
                visualizer.save('{}_judgement.jpg'.format(imgname))
            else:
                visualizer.show('judgement', resize_factor)
        # visualizer.save('judgement.jpg')
        # return visualizer.img
        '''------------------------------------------------------------------------------------------'''
        self.paper = paper

    @func_time
    def grpc(self, img, imgname='test', batch_size=64):
        """
        调用pipeline，对外接口
        :param img:传入原始图像对象
        :param batch_size:
        :return: json串
        """
        start_time = time.time()
        img_h, img_w = img.shape[0], img.shape[1]
        # minlength = min([img_h, img_w])
        # maxlength = max([img_h, img_w])
        self.analyse(img, imgname, batch_size)

        end_time = time.time()

        valid_area = [0, 0, img_w, 0, 0, img_h, img_w, img_h]
        self.result2json(valid_area, [start_time, end_time])

    def result2json(self, valid_area, times):
        valid_areas = list()
        valid_area_pos = dict(zip(["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"], valid_area))
        valid_areas.append(valid_area_pos)
        self.ret["valid_area"] = valid_areas
        timer = dict(zip(["start_time", "end_time"], times))
        self.ret["time"] = timer
        show_results = []
        self.ret["results"] = show_results
        all_results = dict()
        right_results = []
        error_results = []
        problem_results = []
        show_results = []
        self.ret["results"] = show_results
        all_results['right_results'] = right_results
        all_results['error_results'] = error_results
        all_results['problem_results'] = problem_results
        self.ret["all_results"] = all_results
        img_h = valid_area[5]
        img_w = valid_area[2]
        for paperIndex, question in self.paper.paperIndexToQuestion.items():
            res = dict()
            questionbbox = self.paper.paperIndexToBbox[question.indexInPaper]
            # kk: 190709 将输出框限制在原图范围内
            questionbbox.left = max([questionbbox.left, 0])
            questionbbox.top = max([questionbbox.top, 0])
            questionbbox.right = min([questionbbox.right, img_w])
            questionbbox.bottom = min([questionbbox.bottom, img_h])
            res["box"] = [questionbbox.left, questionbbox.top, questionbbox.right, questionbbox.bottom]
            res["text_content"] = question.output
            res["recognition_result"] = question.recognition_result
            if paperIndex in self.paper.allHS:
                res["id"] = 1  # 横式
            elif paperIndex in self.paper.SSWithStem:
                res["id"] = 2  # 带题干竖式
            elif paperIndex in self.paper.waitForStem:
                res["id"] = 3  # 不带题干竖式
            elif paperIndex in self.paper.alreadyHaveStem:
                res["id"] = 4  # 脱式
            else:
                res["id"] = 5  # 未知题型
            if question.state == Result.correct:
                res["ans_result"] = 1
                right_results.append(res)
                show_results.append(res)
            elif question.state == Result.incorrect:
                res["ans_result"] = 0
                error_results.append(res)
                show_results.append(res)
            elif question.state == Result.problem:
                res["ans_result"] = -1
                problem_results.append(res)
            else:
                pass

        if self.aiqa:
            aiqa_all_textline = dict()
            aiqa_all_textline['all_text'] = self.aiqaAllTextLine
            self.ret['aiqa'] = aiqa_all_textline
#
#
if __name__ == "__main__":

    ikkyyu_pipeline = pipeline()
    ikkyyu_pipeline.visual = True
    ikkyyu_pipeline.clock = True
    ikkyyu_pipeline.write = True
    ikkyyu_pipeline.aiqa = True

    # img_path = '/home/yichao/Pictures/71.jpg'
    # for img in os.listdir(img_path):
    #     image = cv2.imread(os.path.join(img_path, img))
    #     try:
    #         ikkyyu_pipeline.analyse(image)
    #     except Exception as e:
    #         print(e)
    #         continue

    # img_path = '/home/yichao/Downloads/err_/00415c4c-3726-11e9-9b17-00163e3060a3.jpg'
    for iteration in range(1):
        img_path = '/workspace/boby/project_git/pipeline_for_ikkyyu/IMG_20181220_110227.jpg'
        image = cv2.imread(img_path)
        # img_h, img_w = image.shape[0], image.shape[1]
        # new_h, new_w = 700, 700*img_w // img_h
        # image = cv2.resize(image, (new_w, new_h))
        # ikkyyu_pipeline.analyse(image)
        ikkyyu_pipeline.grpc(image, batch_size=64)
        print(ikkyyu_pipeline.ret)
