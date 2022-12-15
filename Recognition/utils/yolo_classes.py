

COCO_CLASSES_LIST = [
    'go straight',
    'turn left',
    'turn right',
    'not left',
    'not right'
]


def get_cls_dict(category_num):
    if category_num == 5:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
