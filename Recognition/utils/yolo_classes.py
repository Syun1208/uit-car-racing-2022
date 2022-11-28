

COCO_CLASSES_LIST = [
    'camtrai',
    'camphai',
    'camthang',
    'trai',
    'phai',
    'thang'
]


def get_cls_dict(category_num):
    if category_num == 6:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
