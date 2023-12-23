from numpy.lib.stride_tricks import sliding_window_view

def mkwindows(x, y):
    return sliding_window_view(x, y)

