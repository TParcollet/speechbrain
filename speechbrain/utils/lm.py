"""Language model utilities
"""
import math
import functools

# Define the log base centrally
LOG = math.log10
EXP = functools.partial(math.pow, 10.0)
NEGINFINITY = float("-inf")
