from exception import BboxIndiceError

class Bbox(object):
    def __init__(self, img, indices, BboxType = None, indexInPaper = None):
        self.img = img
        self.eightcoord = []
        img_w, img_h = img.shape[0], img.shape[1]
        if isinstance(indices, str):
            indices = indices.replace('L', '').replace('M', '').replace('Z', '').split(' ')[:8]
            for i in range(8):
                if i % 2 == 0:
                    indices[i] = int(float(indices[i]) * img_w)
                else:
                    indices[i] = int(float(indices[i]) * img_h)
        if len(indices) == 4:
            self.left, self.top, self.right, self.bottom = indices
            self.eightcoord = [self.left, self.top, self.right, self.top, self.right, self.bottom, self.left, self.bottom]
            self.dettype = 0

        elif len(indices) == 8:
            self.top = min(indices[1], indices[3], indices[5], indices[7])
            self.bottom = max(indices[1], indices[3], indices[5], indices[7])
            self.left = min(indices[0], indices[6], indices[2], indices[4])
            self.right = max(indices[0], indices[6], indices[2], indices[4])
            self.eightcoord = indices
            self.dettype = 1
        else:
            raise BboxIndiceError('Bbox坐标数量不为4或8个，请检查')
        self.type = BboxType
        self.indexInPaper = indexInPaper
        self.greedySearchResult = None
        self.beamSearchResult = None
        self.attentionResult = None
    def indexOutput(self):
        img_w, img_h = self.img.shape[0], self.img.shape[1]
        left = self.left / img_w
        right = self.right / img_w
        top = self.top / img_h
        bottom = self.bottom / img_h
        return ' '.join(['M'+str(left), str(top), 'L'+str(right), str(top),
        'L'+str(right), str(bottom), 'L'+str(left), str(bottom), 'Z'])




class BboxType(object):
    HS, SS_child, SS_parent, TS_child, TS_parent, others = range(6)