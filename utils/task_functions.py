def choose_object_at_point(x, y, cats, bboxes, masks):
    """
    returns index of detected object at given image point, smallest one in case of multiple objects at point
    :param x: x (horizontal) point value
    :param y: y (vertical) point value
    :param cats: np.array of detected objects categories
    :param bboxes: np.array of detected objects bounding boxes
    :param masks: np.array of detected objects masks
    :return: index of smallest object at point from array of detected objects (e.g. self.mask),
             None if no objects at point
    """
    all_objects_at_point_idx = []
    for i in range(len(cats)):
        if masks[i][y][x]:
            all_objects_at_point_idx.append(i)
    if len(all_objects_at_point_idx) == 0:
        return None
    elif len(all_objects_at_point_idx) == 1:
        object_at_point_idx = all_objects_at_point_idx[0]
    else:
        w = bboxes[all_objects_at_point_idx[0]][2] - bboxes[all_objects_at_point_idx[0]][0]
        h = bboxes[all_objects_at_point_idx[0]][3] - bboxes[all_objects_at_point_idx[0]][1]
        smallest_object_size = w * h
        object_at_point_idx = all_objects_at_point_idx[0]
        for i in range(1, len(all_objects_at_point_idx)):
            w = bboxes[all_objects_at_point_idx[i]][2] - bboxes[all_objects_at_point_idx[i]][0]
            h = bboxes[all_objects_at_point_idx[i]][3] - bboxes[all_objects_at_point_idx[i]][1]
            object_size = w * h
            if object_size < smallest_object_size:
                object_at_point_idx = i
    return object_at_point_idx


def is_point_reachable(scene, point):
    """
    checks if robot can reach given point
    """

    def is_point_floor(point):
        """
        checks if point height is within range of expected floor
        """
        floor_height = 15
        floor_height_threshold = 5
        if abs(point[1] - floor_height) < floor_height_threshold:
            return True
        else:
            return False

    def is_area_clear(scene, point):
        """
        checks if there are no points detected within specified area over given point
        """
        horizontal_radius_restricted = 300
        height_restricted_min = 50
        height_restricted_max = 500

        for row in scene:
            for scene_point in row:
                distance_horizontal = ((scene_point[0] - point[0]) ** 2 + (scene_point[2] - point[2]) ** 2) ** 0.5
                if abs(distance_horizontal) < horizontal_radius_restricted:
                    distance_vertical = scene_point[1] - point[1]
                    if height_restricted_min < distance_vertical < height_restricted_max:
                        return False
        return True

    if is_point_floor(point):
        if is_area_clear(scene, point):
            return True
    return False

