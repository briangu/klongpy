
def nilad():
    return "hello, world!"

def monad(x):
    return f"{x}+1"

def dyad(x, y):
    return f"{x}*{y}+1"

def triad(x, y, z):
    return f"{x}*{y}+{z}+1"

def knilad(klong):
    return klong("2+2")

def kmonad(klong, x):
    return klong(f"{x}+1")

def kdyad(klong, x, y):
    return klong(f"{x}*{y}+1")

def ktriad(klong, x, y, z):
    return klong(f"{x}*{y}+{z}+1")

def not_exported():
    raise RuntimeError()
