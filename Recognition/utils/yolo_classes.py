

COCO_CLASSES_LIST = [
    'end20',
    'start20',
    'thang',
    'trai',
    'phai',
    'stop'
]


def get_cls_dict(category_num):
    if category_num == 6:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
