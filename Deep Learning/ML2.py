import numpy as np
# 1-Dimensional Array
# Example: 1 = [2, 3, 4, 5]
#
# Shape - Represents the size of each dimension
# Shape of example array: (4, )
#
# Type:
# 1-D array
# In mathematics, this is a vector

# 2-Dimensional Array
# Example: 
# lol = [[1, 2, 3, 4],
#        [1, 2, 3, 4]]
#
# Shape: (2, 4) - 2-Dimensional array with four elements in each row
# NOTE: The array must be homologous meaning each dimension needs have the same size
#
# Type: 2-D array
# In mathematics, this is a matrix

# The Dot Product
i = [1, 2, 3, 2.5]          # Inputs 
w = [0.2, 0.8, -0.5, 1.0]   # Weights
b = 2                       # Bias

# Dot Product Function 
def dot_product(v1: list, v2: list) -> float:
    result = 0

    assert(len(v1) == len(v2))

    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result
 
# Calculating Dot Product
o = np.dot(w, i) + b        # Output
# out = dot_product(w, i) + b
print(o) # 4.8
