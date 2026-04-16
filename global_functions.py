import numpy as np

import global_params as params

def get_s1(A1, t):
    return A1 * np.sin(2*np.pi * params.f * t)

def get_s2(A2, t):
    return A2 * np.sin(2*np.pi * (params.f+params.df) * t)

