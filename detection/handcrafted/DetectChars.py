import math


# const check-if-possible-char
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# const compare-2-chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# const others
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100



def check_possible_char(possible_char):
    """
    "first" raw check contour could be char
    :param possible_char:
    :return: True/False state
    """

    if possible_char.box_area > MIN_CONTOUR_AREA \
            and possible_char.box_w > MIN_PIXEL_WIDTH \
            and possible_char.box_h > MIN_PIXEL_HEIGHT \
            and MIN_ASPECT_RATIO < possible_char.box_aspect_ratio < MAX_ASPECT_RATIO:
        return True
    else:
        return False


def find_list_of_list_matching_chars(list_possible_chars):
    """
    start with the possible chars of one big list -> re-arrange the one big list of chars into a list of lists of matching chars
    :param list_possible_chars:
    :return: list_of_list_matching_chars [list[Char1, Char2], list[Char1, Char2],...]
    """

    list_of_list_matching_chars = []  # return value

    for possible_char in list_possible_chars:
        # find all char in big list that match the current char
        list_matching_chars = find_list_matching_chars(possible_char, list_possible_chars)
        # also add the current char to current possible list of matching chars
        list_matching_chars.append(possible_char)

        # check of list of matching char is not long enough to constitute a plate -> pass
        if len(list_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        # add to list of list of matching chars
        list_of_list_matching_chars.append(list_matching_chars)

        # remove the current list of matching char from big list -> dont use those same chars twice
        # make sure to make a new big list since dont want to change the original big list
        list_of_possible_chars_with_current_match_removed = list(set(list_possible_chars) - set(list_matching_chars))

        # recursive call
        recursive_list_of_list_of_matching_chars = find_list_of_list_matching_chars(
            list_of_possible_chars_with_current_match_removed)

        # for each list of matching chars found by recursive call
        # add to original list of list of matching chars
        for recursive_list_of_matching_chars in recursive_list_of_list_of_matching_chars:
            list_of_list_matching_chars.append(recursive_list_of_matching_chars)

        break

    return list_of_list_matching_chars


def find_list_matching_chars(possible_char, list_of_chars):
    """
    given a possible char and big list of possible char
    -> find all chars in list that match with this-possible-char
    :param possible_char, list_of_chars
    :param list_of_chars: all chars match with given possible chars
    :return:
    """

    list_matching_chars = []  # return value

    for possible_matching_char in list_of_chars:
        if possible_matching_char == possible_char:
            continue

        # compute stuff to see if chars are a match
        distance_2chars = distance_between_chars(possible_char, possible_matching_char)
        angle_2chars = angle_between_chars(possible_char, possible_matching_char)
        change_in_area = abs(possible_char.box_area - possible_matching_char.box_area) / possible_char.box_area
        change_in_width = abs(possible_char.box_w - possible_matching_char.box_w) / possible_char.box_w
        change_in_height = abs(possible_char.box_h - possible_matching_char.box_h) / possible_char.box_h

        # check if chars match
        if distance_2chars < possible_char.box_diagonal * MAX_DIAG_SIZE_MULTIPLE_AWAY \
                and angle_2chars < MAX_ANGLE_BETWEEN_CHARS \
                and change_in_area < MAX_CHANGE_IN_AREA \
                and change_in_width < MAX_CHANGE_IN_WIDTH \
                and change_in_height < MAX_CHANGE_IN_HEIGHT:
            list_matching_chars.append(possible_matching_char)

    return list_matching_chars


def distance_between_chars(char1, char2):
    # distance between 2 chars
    x = abs(char1.box_centerX - char2.box_centerX)
    y = abs(char1.box_centerY - char2.box_centerY)
    return math.sqrt(x ** 2 + y ** 2)


def angle_between_chars(char1, char2):
    # angle between 2 chars
    distance_horizontal = float(abs(char1.box_centerX - char2.box_centerX))
    distance_vertical = float(abs(char1.box_centerY - char2.box_centerY))

    if distance_horizontal != 0.0:
        angle = math.atan(distance_vertical / distance_horizontal)
    else:
        angle = 1.5708  # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    angle_in_degree = angle * (180.0 / math.pi)  # calculate angle in degrees
    return angle_in_degree

