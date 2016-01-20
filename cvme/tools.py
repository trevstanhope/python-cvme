def maprange(val, a, b):
    (a1, a2), (b1, b2) = a, b
    return  b1 + ((val - a1) * (b2 - b1) / (a2 - a1))
