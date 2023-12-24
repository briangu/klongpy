from numpy.lib.stride_tricks import sliding_window_view

# Create a sliding window view of the data using the very efficient stride_tricks
# TODO: maybe this should be part of the core language?
def mkwindows(x, y):
    return sliding_window_view(x, y)

