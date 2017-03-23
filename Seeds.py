from point import Point


def Seeds(nRows: int, nCols: int, Row_num: int, Col_num: int, Row_step: int, Col_step: int, seed_num: int,
          point_array: list):
    Row_remain = nRows - Row_step * Row_num
    Col_remain = nCols - Col_step * Col_num
    t1 = 1
    t2 = 1
    count = 0
    centerx = -1
    centery = -1
    for i in range(Row_num):
        t2 = 1
        for j in range(Col_num):
            centerx = int(i * Row_step + 0.5 * Row_step + t1)
            centery = int(j * Col_step + 0.5 * Col_step + t2)
            centerx = nRows - 1 if (centerx >= nRows - 1) else centerx
            centery = nCols - 1 if (centery >= nCols - 1) else centery
            if t2 < Col_remain:
                t2 += 1
            point_array.append(Point(centerx, centery))
            count += 1
        if t1 < Row_remain:
            t1 += 1
    return count
