from point import Point
import math


# def Seeds_deprecated(nRows: int, nCols: int, Row_num: int, Col_num: int, Row_step: int, Col_step: int, seed_num: int,
#                      point_array: list):
#     Row_remain = nRows - Row_step * Row_num
#     Col_remain = nCols - Col_step * Col_num
#     t1 = 1
#     t2 = 1
#     count = 0
#     centerx = -1
#     centery = -1
#     for i in range(Row_num):
#         t2 = 1
#         for j in range(Col_num):
#             centerx = int(i * Row_step + 0.5 * Row_step + t1)
#             centery = int(j * Col_step + 0.5 * Col_step + t2)
#             centerx = nRows - 1 if (centerx >= nRows - 1) else centerx
#             centery = nCols - 1 if (centery >= nCols - 1) else centery
#             if t2 < Col_remain:
#                 t2 += 1
#             point_array.append(Point(centerx, centery))
#             count += 1
#         if t1 < Row_remain:
#             t1 += 1
#     return count


def gen_seeds(row_num: int, col_num: int, seed_num: int) -> tuple:
    """
    generate seeds
    :param row_num: the image row number
    :param col_num: the image column number
    :param seed_num: the seed number should be generated
    :return:
        seeds_list: a tuple contains all the seeds
    """
    seeds_list = []
    num_seeds_col = int(math.sqrt(seed_num * col_num / row_num))
    num_seeds_row = int(seed_num / num_seeds_col)
    step_x = int(row_num / num_seeds_row)
    step_y = int(col_num / num_seeds_col)
    row_remain = row_num - num_seeds_row * step_x
    col_remain = col_num - num_seeds_col * step_y
    current_row_remain = 1
    current_col_remain = 1
    current_seeds_count = 0
    for i in range(num_seeds_row):
        for j in range(num_seeds_col):
            center_x = int(i * step_x + 0.5 * step_x + current_row_remain)
            center_y = int(j * step_y + 0.5 * step_y + current_col_remain)
            if center_x > row_num - 1:
                center_x = row_num - 1
            if center_y > col_num - 1:
                center_y = col_num - 1
            if current_col_remain < col_remain:
                current_col_remain += 1
            seeds_list.append(Point(x=center_x, y=center_y))
            current_seeds_count += 1
        if current_row_remain < row_remain:
            current_row_remain += 1
    return tuple(seeds_list)
