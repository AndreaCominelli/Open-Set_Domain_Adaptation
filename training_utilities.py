# log uniform
# https://en.wikipedia.org/wiki/Reciprocal_distribution

import math
from scipy.stats import expon

print(list(expon(scale = 100)))