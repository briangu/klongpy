from .backend import np

def add(a, b):
    return np.add(a, b)

def subtract(a, b):
    return np.subtract(a, b)

def multiply(a, b):
    return np.multiply(a, b)

def divide(a, b):
    return np.divide(a, b)

def map(a, func):
    return np.vectorize(func)(a)

def set_dtype(dtype):
    return None

def get_dtype():
    return "f64"
