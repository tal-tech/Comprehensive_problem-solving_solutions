# encoding: utf-8
import ocr_beam.det_beam_search as det_beam_search
import cv2
from recognition_config import Config as config
import re
def image_normalization(image):
    if image.shape[0] != 32:
        image = cv2.resize(image, (int(image.shape[1] / image.shape[0] * 32), 32))
    if image.shape[1] < 10:
        image = cv2.resize(image, (10, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255 * 2 - 1
    return image

def label_replace_for_error(label):
    '''
    将label里面的非法字符转换为合法字符
    :param label:
    :return:
    '''

    # 题号替换
    label = label.replace(' ', ' ')
    if ' ' in label:
        label_list = label.split(' ')
        if label_list[1] and label_list[0] in config.QUESTION_NUM_REPLACE:
            label_list[0] = '○' + label_list[0] + '○'
        label = ''.join(label_list)

    label = label.replace('（', '(')
    label = label.replace('）', ')')
    label = label.replace('４', '4')
    label = label.replace('１', '1')
    label = label.replace('５', '5')
    label = label.replace('８', '8')
    label = label.replace('９', '9')
    label = label.replace('＋', '+')
    label = label.replace('２', '2')
    label = label.replace('０', '0')
    label = label.replace('６', '6')
    label = label.replace('３', '3')
    label = label.replace('７', '7')
    label = label.replace('－', '-')
    label = label.replace('　', '')
    label = label.replace('？', '?')
    label = label.replace('，', ',')
    label = label.replace('：', ':')
    label = label.replace('＞', '>')
    label = label.replace('＜', '<')
    label = label.replace('﹥', '>')
    label = label.replace('﹤', '<')
    label = label.replace('！', '!')
    label = label.replace('＝', '=')
    label = label.replace('—', '~')
    label = label.replace('√', '')
    label = label.replace(' ', '')
    label = label.replace('＇', "'")
    label = label.replace('⑴', '(1)')
    label = label.replace('⑵', '(2)')
    label = label.replace('⑶', '(3)')
    label = label.replace('⑷', '(4)')
    label = label.replace('⑸', '(5)')
    label = label.replace('⑹', '(6)')
    label = label.replace('⑺', '(7)')
    label = label.replace('⑻', '(8)')
    label = label.replace('⑼', '(9)')
    label = label.replace('⑽', '(10)')
    label = label.replace('⑾', '(11)')
    label = label.replace('⑿', '(12)')
    label = label.replace('⒀', '(13)')
    label = label.replace('⒁', '(14)')
    label = label.replace('⒂', '(15)')
    label = label.replace('⒃', '(16)')
    label = label.replace('⒄', '(17)')
    label = label.replace('⒅', '(18)')
    label = label.replace('⒆', '(19)')
    label = label.replace('⒇', '(20)')
    label = label.replace('_', '')
    label = label.replace('/', '|')
    return label


def label_replace_for_train(label):
    '''
    针对不同的训练方案提出的替换方式
    :param label:
    :return:
    '''

    label = re.sub('○.*○', '○', label)
    question = set(config.QUESTION_NUM) & set(label)
    if question:
        label = label.replace(list(question)[0], '○')
    label = label.replace('$', '')
    label = label.replace('#', '')

    return label

