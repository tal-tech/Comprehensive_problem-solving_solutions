from character_dict import character_dict

class Config(object):
    TRAIN_DATA_LIST = ['/workspace/newpipetraining/enhance','/workspace/train/market/enhance', '/workspace/train/toC/enhance', '/workspace/train/toC_7/enhance', '/workspace/train/toC_7f/enhance', '/workspace/train/toC_8/enhance']
    TRAIN_DATA_BATCH = [1, 1, 1, 1, 60, 1]
    TRAIN_DATA_NAME = ['3351_3352', 'market', 'toC', 'toC_7', 'toC_7f', 'toC_8']
    VAL_DATA_LIST = ['/workspace/newpipevalidation/orgin','/workspace/val/market/orgin', '/workspace/val/toC/orgin', '/workspace/val/toC_7/orgin', '/workspace/val/toC_7f/orgin', '/workspace/val/toC_8/orgin']
    VAL_DATA_BATCH = [3, 3, 3, 3, 24, 3]
    VAL_DATA_NAME = ['3351_3352','market','toC','toC_7','toC_7f','toC_8']
    CRNN_IMG_CONFIG = {'normalization': True, 'w_stable': False, 'height': 32, 'w_max': 250, 'w_min': 32}
    QUESTION_NUM = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿'
    QUESTION_NUM_REPLACE = ['51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65',
                            '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
                            '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95',
                            '96', '97', '98', '99', '100']
    DATA_ENHANCE = False
    IMAGE_HEIGHT = 32
    BATCH_SIZE = 32




    RNN_UNITS = 256
    LEARN_RATE = 1e-4
    MODEL_SAVE = 'ikkyyu_recognition/crnn/237999.ckpt'
    DECODE = '10853-2=6×7÷49+()*@~○><{}|x.千米分平方公毫立升吨克元角日时秒厘 '#空格表示空
    def __init__(self): #构建字符集
        self.ONE_HOT = dict()
        self.NUM_SIGN = list()
        characters = character_dict()
        subdict_list = [member for member in dir(characters) if '__' not in member]
        for subdict in subdict_list:
            self.ONE_HOT.update(eval(''.join(['characters.', subdict])))
        for char in self.DECODE:
            if char!= ' ':
                self.NUM_SIGN.append(self.ONE_HOT[char])
        self.NUM_SIGN.append(len(self.ONE_HOT))

if __name__ == '__main__':
    c = Config()
