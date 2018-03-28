# -*- coding: UTF-8 -*-
import numpy as np

len = 10
count = 4

def encode(text):
    vector = np.zeros(count*len, dtype=float)
    for i, c in enumerate(text):
        vector[i * len + int(c)] = 1
    return vector

def decode(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_idx = c % len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        text.append(chr(char_code))
    return "".join(text)

