import numpy as np

# Dot Product of a layer of neurons
i = [1, 2, 3, 2.5] # Vector

# Matrix 
w = [[0.2, 0.8, -0.5, 1.0],         # First Neuron
     [0.5, -0.91, 0.26, -0.5],      # Second Neuron
     [-0.26, -0.27, 0.17, 0.87]]    # Third Neuron
b = [2, 3, 0.5] # Corresponding biases

# Matrix Dot Product
def matrix_dot(m1, v1):
    res = []
    sum = 0

    for row in m1:
        for i in range(len(v1)):
            sum += row[i] * v1[i]

        res.append(sum)
        sum = 0

    return res


o = np.dot(w, i) + b
print(o)

