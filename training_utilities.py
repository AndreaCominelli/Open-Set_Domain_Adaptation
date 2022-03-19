# log uniform
# https://en.wikipedia.org/wiki/Reciprocal_distribution

import math
import numpy as npy

def log_uniform(a, b):
    pdfs = []
    range = npy.linspace(a, b)
    for x in list(range):
        f = 1 / (x * (math.log(b) - math.log(a)))
        pdfs.append(f)
    return pdfs

if __name__ == "__main__":
    a = 1 * 10 ** -4
    b = 1 * 10 ** -1
    res = log_uniform(a, b)
    print(res)