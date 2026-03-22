import math

def log_transform(values):
    return [math.log1p(v) for v in values]