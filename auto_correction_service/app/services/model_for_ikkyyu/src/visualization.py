import cv2
from bbox import Bbox
from PIL import Image, ImageDraw, ImageFont
from util import OpenCVtoPIL, PILtoOpenCV
import os

class visualization(object):
    def __init__(self, img):
        self.origin_img = img
        self.img = img.copy()
        font_path = os.path.dirname(__file__)+'/../Fonts/fzjl.ttf'
        self.font = ImageFont.truetype(font_path, 30)
        # self.font = ImageFont.truetype('Fonts/fzjl.ttf', 30)


    def show(self, windowName = 'unknown', resize = 1):
        if resize != 1:
            img_h, img_w = self.img.shape[0], self.img.shape[1]
            cv2.imshow(windowName, cv2.resize(self.img, (round(img_w * resize), round(img_h * resize))))
        else:
            cv2.imshow(windowName, self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def drawBbox(self, Bbox, color = (255, 0, 0)):
        if Bbox.eightcoord:
            cv2.line(self.img, (int(Bbox.eightcoord[0]), int(Bbox.eightcoord[1])), (int(Bbox.eightcoord[2]), int(Bbox.eightcoord[3])), color, 2)
            cv2.line(self.img, (int(Bbox.eightcoord[2]), int(Bbox.eightcoord[3])), (int(Bbox.eightcoord[4]), int(Bbox.eightcoord[5])), color, 2)
            cv2.line(self.img, (int(Bbox.eightcoord[4]), int(Bbox.eightcoord[5])), (int(Bbox.eightcoord[6]), int(Bbox.eightcoord[7])), color, 2)
            cv2.line(self.img, (int(Bbox.eightcoord[6]), int(Bbox.eightcoord[7])), (int(Bbox.eightcoord[0]), int(Bbox.eightcoord[1])), color, 2)
        else:
            cv2.rectangle(self.img, (Bbox.left, Bbox.top), (Bbox.right, Bbox.bottom), color, 2)

    def text(self, Bbox, text, font=None, fillColor=(255, 0, 0)):
        if not font:
            font = self.font
        position = (Bbox.left + int((Bbox.right - Bbox.left)/4), Bbox.top + int((Bbox.bottom - Bbox.top)/4))
        img_PIL = OpenCVtoPIL(self.img)
        draw = ImageDraw.Draw(img_PIL)
        if text:
            draw.text(position, text, font=font, fill=fillColor)
        self.img = PILtoOpenCV(img_PIL)

    def save(self, output_name='output.jpg'):
        path="./test_result/"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+output_name, self.img)


    def reset(self):
        self.img = self.origin_img.copy()



if __name__ == "__main__":
    import json
    img = cv2.imread('/home/yichao/Documents/CC_work/new_ikkyyu/detection_output/inf2.jpg')
    with open('/home/yichao/Documents/CC_work/new_ikkyyu/detection_output/infer_2 (2).json') as json_file:
        js = json.load(json_file)
    visualizer = visualization(img)
    for bbox in js:
        if len(bbox['children']) != 0:
            for child in bbox['children']:
                visualizer.drawBbox(Bbox(img, child))
        visualizer.drawBbox(Bbox(img, bbox['parent']), color = (0, 255, 0))
    visualizer.show('test', 0.5)


    

