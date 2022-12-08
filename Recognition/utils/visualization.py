


import numpy as np
import cv2


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: tổng số class

    # Output
      bgrs: a list of (B, G, R) tuples
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """

    # Arguments
      img: the input image as a numpy array.
      text: the text để vẽ.
      topleft: góc trái của boudingbox để vẽ.
      color: color of the patch.

   
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0]) 
    h = min(h, img_h - topleft[1])

    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: danh sách tên các class.
    """

    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))

    def draw_bboxes(self, img, boxes, confs, clss,session,inputname):
        """vẽ bounding boxes."""
        centers=[]
        cll=-1
        prediction=-1
        area=-1
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            cll=cl
            
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            if cl == 2 or cl ==3 or cl == 4:          #nếu là đi thẳng, quẹo trái, quẹo phải thì phải kiểm trả 1 model CNN nữa để nhận diện
              X_test=img[y_min:y_max,x_min:x_max]
              X_test=cv2.resize(X_test,(30, 30))
              X_test = X_test.astype('float32')/255
              X_test = X_test.reshape(1,30,30,3)
              
              prediction = session.run(None,{inputname:X_test})
              prediction = np.squeeze(prediction) 
              cl =np.argmax(prediction)+2
              # cl = int(cl)
              cll=cl
            color = self.colors[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            txt = '{} {:.2f}'.format(cls_name, cf)
            img = draw_boxed_text(img, txt, txt_loc, color)
            c=y_max-y_min
            d=x_max-x_min
            area=c*d
            # if (c/d<1.5 or d/c <1.5)and c*d >6 and c*d <40000:
            x=int((x_min+ x_max)/2.0)
            y=int((y_min+y_max)/2.0)
            b = np.array([[x], [y]])
            centers.append(np.round(b))
            # return img,centers,cl
        print(area)
        return img,centers,cll,area
        
