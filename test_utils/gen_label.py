def gen_label(nRows: int, nCols: int, default_value=-1, gen_function=None):
    if gen_function is not None:
        return [gen_function(_) for _ in range(nRows * nCols)]
    return [default_value for _ in range(nRows * nCols)]
