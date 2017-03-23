class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "<class 'Point'>: x={}, y={}".format(self.x, self.y)

    __repr__ = __str__
