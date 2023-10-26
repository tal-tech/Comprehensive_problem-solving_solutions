def is_contain(coord1, coord2):
    if coord1[0] >= coord2[0] and coord1[1] >= coord2[1] and coord1[2] <= coord2[2] and coord1[3] <= coord2[3]:
        return True
    return False

def bboxes_area(bboxes):
    h = bboxes[2] - bboxes[0]
    w = bboxes[3] - bboxes[1]
    return h*w

def IOU_calculation(bbox1, bbox2):
    """
    bbox1 : 小框
    bbox2 : 大框
    """
    box_interaction = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
                       min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])]
    interaction_width= box_interaction[2] - box_interaction[0]
    interaction_height = box_interaction[3] - box_interaction[1]
    if interaction_width > 0 and interaction_height > 0:
        interaction_area = interaction_height * interaction_width
        union_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + \
                    (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - interaction_area
        return interaction_area / union_area
    else:
        return 0
    
    
def IOS_calculation(bbox1, bbox2):
    """
    bbox1 : 小框
    bbox2 : 大框
    """
    area1 = bboxes_area(bbox1)
    area2 = bboxes_area(bbox2)
    box_intersection = [max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
                       min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])]
    intersection_width = box_intersection[2] - box_intersection[0]
    intersection_height = box_intersection[3] - box_intersection[1]
    if intersection_width > 0 and intersection_height > 0:
        intersection_area = intersection_width * intersection_height
        return intersection_area / min(area1, area2)
    else:
        return 0