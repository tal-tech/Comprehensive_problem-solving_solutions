from bbox import Bbox, BboxType
from util import image_normalization, boundingBox, italic_to_rec, get_legal_level, isdigit, expansion_box
import numpy as np
import math
from question import Question
import re
import cv2

class Paper(object):
    def __init__(self, img):
        self.recognitionIndexToBbox = dict()  # 根据识别序号找bbox
        self.paperIndexToRecognitionIndex = dict()  # 根据版面序号找识别序号
        self.recognitionImgList = list()  # 存放归一化好的小图list
        self.recognitionExImgList = list()  # 存放归一化好的小图list
        self.paperIndexToBbox = dict()  # 根据版面序号找bbox
        self.nextBboxIndex = 0
        self.nextRecognitionIndex = 0
        self.maxWide = 0
        self.img = img
        self.parentToChildren = dict()  # parent, children的版面序号映射关系
        self.waitForStem = set()  # 竖式大框版面序号，要找对应题干。
        self.alreadyHaveStem = set()  # 脱式大框版面序号，无需找对应题干。
        self.SSWithStem = set()  # 匹配到题干的竖式版面序号。
        self.allHS = set()  # 所有横式的版面序号，其中包括未经匹配的竖式题干。经过竖式题干匹配后与waitForStem的并集为全部题目。
        self.paperIndexToQuestion = dict()

    def loadBboxesfromjson(self, detection_json, expansion_rate = 0):
        def addToRecognitionQueue(bbox_coordinate, bbox_type):
            bbox = Bbox(self.img, bbox_coordinate, bbox_type, self.nextBboxIndex)
            self.recognitionIndexToBbox[self.nextRecognitionIndex] = bbox
            self.paperIndexToRecognitionIndex[bbox.indexInPaper] = self.nextRecognitionIndex
            self.nextBboxIndex += 1
            self.nextRecognitionIndex += 1
            img_h, img_w = self.img.shape[0], self.img.shape[1]
            expansion_width = round(expansion_rate * img_w)
            expansion_height = round(expansion_rate * img_h)

            if bbox.dettype == 1:
                cropImg = italic_to_rec(self.img, bbox.eightcoord)
                expansionImg = italic_to_rec(self.img, expansion_box(bbox.eightcoord))
            else:
                cropImg = self.img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
                expansionImg = self.img[max(bbox.top - expansion_height, 0): min(bbox.bottom + expansion_height, img_h),
                          max(bbox.left - expansion_width, 0): min(bbox.right + expansion_width, img_w)]

            cropImg_normalization = image_normalization(cropImg)
            expansionImg_normalization = image_normalization(expansionImg)

            self.maxWide = max(self.maxWide, cropImg_normalization.shape[1])
            # self.recognitionImgList.append(cropImg_normalization)
            self.recognitionImgList.append(cropImg)
            self.recognitionExImgList.append(expansionImg_normalization)
            return bbox

        for question in detection_json:
            if question['label'] == 1:
                HSBbox = addToRecognitionQueue(question['parent'], BboxType.HS)
                self.paperIndexToBbox[HSBbox.indexInPaper] = HSBbox
                self.allHS.add(HSBbox.indexInPaper)
            elif question['label'] == 2:
                parentBbox = Bbox(self.img, question['parent'], BboxType.SS_parent, self.nextBboxIndex)
                self.paperIndexToBbox[parentBbox.indexInPaper] = parentBbox
                self.parentToChildren[parentBbox.indexInPaper] = list()
                self.waitForStem.add(parentBbox.indexInPaper)
                self.nextBboxIndex += 1
                for child in question['children']:
                    childBbox = addToRecognitionQueue(child, BboxType.SS_child)
                    self.paperIndexToBbox[childBbox.indexInPaper] = childBbox
                    self.parentToChildren[parentBbox.indexInPaper].append(childBbox.indexInPaper)
            else:
                parentBbox = Bbox(self.img, question['parent'], BboxType.TS_parent, self.nextBboxIndex)
                self.paperIndexToBbox[parentBbox.indexInPaper] = parentBbox
                self.parentToChildren[parentBbox.indexInPaper] = list()
                self.alreadyHaveStem.add(parentBbox.indexInPaper)
                self.nextBboxIndex += 1
                for child in question['children']:
                    childBbox = addToRecognitionQueue(child, BboxType.TS_child)
                    self.paperIndexToBbox[childBbox.indexInPaper] = childBbox
                    self.parentToChildren[parentBbox.indexInPaper].append(childBbox.indexInPaper)

    def createRecognitionInput(self):
        images = np.zeros([self.nextRecognitionIndex, 32, self.maxWide])
        widths = []
        for index, image in enumerate(self.recognitionImgList):
            image = image_normalization(image)
            images[index, :, 0: image.shape[1]] = image
            widths.append(image.shape[1])
        images = images[..., np.newaxis]
        widths = np.array(widths, dtype=np.int32)
        return images, widths

    def fillInGreedySearchResult(self, output_list):
        for index, output in enumerate(output_list):
            self.recognitionIndexToBbox[index].greedySearchResult = output


    def fillInBeamSearchResult(self, bbox_index_list, output_list, conf_vec):
        for index, output in enumerate(output_list):
            self.paperIndexToBbox[bbox_index_list[index]].beamSearchResult = output

    def removeZerosandDot(self, string):
        string = string.lstrip('0')
        string = string.rstrip('0')
        string = string.replace('.','')
        return string

    def findStemForSS(self, distanceThreshold, similarityThreshold):
        HSdigitsDict = dict()  # 横式编号-->与其所包含的数字集合

        def generateSetFromString(string):
            numSet = set(re.findall(r'\d+\.?\d*', string))
            oldNumList = []
            newNumList = []
            for oldNum in numSet:
                newNum = self.removeZerosandDot(oldNum)
                if newNum != oldNum:
                    oldNumList.append(oldNum)
                    newNumList.append(newNum)
            for i in range(len(oldNumList)):
                numSet.remove(oldNumList[i])
                numSet.add(newNumList[i])
            return numSet

        def similarity(HSBbox, SSBbox, HSdigitsSet, SSdigitsSet):
            ##############################
            ######认为有点问题,待修改######
            ##############################
            # kk:修改了竖式搜索题干时的距离策略, 待大量测试
            if not HSdigitsSet:
                return 0
            HSVerticalCenter = (HSBbox.top + HSBbox.bottom) / 2
            HSHorizontalCenter = (HSBbox.left + HSBbox.right) / 2
            SSVerticalCenter = (SSBbox.top + SSBbox.bottom) / 2
            SSHorizontalCenter = (SSBbox.left + SSBbox.right) / 2
            if HSBbox.top > SSBbox.bottom: # kk:只找空间位置在竖式水平线及以上的横式
                return 0
            Distance = math.sqrt(pow(HSVerticalCenter - SSVerticalCenter, 2) + pow(HSHorizontalCenter - SSHorizontalCenter, 2)) \
                       - math.sqrt(pow(SSBbox.top - SSVerticalCenter, 2) + pow(SSBbox.left - SSHorizontalCenter, 2))

            if (Distance / self.img.shape[0]) > distanceThreshold:
                return 0
            # minimumVerticalDistance = max(max(HSBbox.top, SSBbox.top) - min(HSBbox.bottom, SSBbox.bottom), 0)
            # minimumHorizontalDistance = max(max(HSBbox.left, SSBbox.left) - min(HSBbox.right, SSBbox.right), 0)
            # minimumDistance = minimumVerticalDistance / self.img.shape[0] + minimumHorizontalDistance / self.img.shape[1]
            # if minimumDistance > distanceThreshold:
            #     return 0

            return len(HSdigitsSet & SSdigitsSet) / len(HSdigitsSet)

        for HSIndex in self.allHS:
            HSdigitsDict[HSIndex] = generateSetFromString(self.paperIndexToBbox[HSIndex].greedySearchResult)
        pairedSSIndexDict = dict()
        for SSIndex in self.waitForStem:
            SSBbox = self.paperIndexToBbox[SSIndex]
            SSdigitsSet = set()
            for childIndex in self.parentToChildren[SSIndex]:
                if isdigit(self.paperIndexToBbox[childIndex].greedySearchResult):
                    SSdigitsSet.update(generateSetFromString(self.paperIndexToBbox[childIndex].greedySearchResult))
                elif len(self.paperIndexToBbox[childIndex].greedySearchResult) > 1 and isdigit(self.paperIndexToBbox[childIndex].greedySearchResult[1:]):
                    SSdigitsSet.update(generateSetFromString(self.paperIndexToBbox[childIndex].greedySearchResult))
                elif '@' in self.paperIndexToBbox[childIndex].greedySearchResult:
                    SSdigitsSet.update(generateSetFromString(self.paperIndexToBbox[childIndex].greedySearchResult))
            # 找与竖式匹配的横式并更新相关字典，集合
            bestSimilarity, bestPairHSIndex = 0, None
            for HSIndex in self.allHS:
                if (not set('{|}') & set(self.paperIndexToBbox[HSIndex].greedySearchResult)) \
                        and (get_legal_level(self.paperIndexToBbox[HSIndex].greedySearchResult) > 0): #kk:找没有分数的且合法(1+1,1+1=,1+1=2)的横式
                    HSBbox = self.paperIndexToBbox[HSIndex]
                    if len(HSdigitsDict[HSIndex]) < 2:
                        continue

                    pairSimilarity = similarity(HSBbox, SSBbox, HSdigitsDict[HSIndex], SSdigitsSet)
                    # print(HSBbox.greedySearchResult, SSdigitsSet, pairSimilarity)
                    if pairSimilarity < similarityThreshold:
                        continue
                    if pairSimilarity > bestSimilarity:
                        bestSimilarity = pairSimilarity
                        bestPairHSIndex = HSIndex
            if bestPairHSIndex:  # 找到了最匹配题干
                pairedSSIndexDict[SSIndex] = bestPairHSIndex
        for SSIndex, HSIndex in pairedSSIndexDict.items():
            if HSIndex in self.allHS:
                self.allHS.remove(HSIndex)  # 从横式中移除
            self.waitForStem.remove(SSIndex)  # 从待查找题干竖式中移除
            # 构造新的大框，将原有parent-children映射关系修改为新的映射关系，并将新的合并框加入已有题干的竖式框集合。
            oldChildren = self.parentToChildren[SSIndex]
            self.parentToChildren.pop(SSIndex)
            oldChildren.append(HSIndex)
            newParrent = Bbox(
                self.img,
                boundingBox(self.paperIndexToBbox[HSIndex], self.paperIndexToBbox[SSIndex]),
                BboxType.SS_parent, self.nextBboxIndex)
            self.parentToChildren[newParrent.indexInPaper] = oldChildren
            self.paperIndexToBbox[newParrent.indexInPaper] = newParrent
            self.nextBboxIndex += 1
            self.SSWithStem.add(newParrent.indexInPaper)


