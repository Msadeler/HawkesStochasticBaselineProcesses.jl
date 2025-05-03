using PyCall
using Distributions
using Random


py"""
import numpy as np
np.random.seed(0)
def simu():
    return(np.random.exponential())
"""
rand(Exponential())

py"simu"()