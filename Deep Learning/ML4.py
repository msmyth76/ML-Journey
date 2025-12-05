import numpy as np 
# Batches - training examples 
i = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
print(np.shape(i))

# First Layer 
w = [[0.2, 0.8, -0.5, 1.0],         # First Neuron
     [0.5, -0.91, 0.26, -0.5],      # Second Neuron
     [-0.26, -0.27, 0.17, 0.87]]    # Third Neuron
print(np.shape(w))

b = [2, 3, 0.5]
print(np.shape(b))

# Second Layer
w2 = [[0.1, -0.14, 0.5],         # First Neuron
     [-0.5, 0.12, -0.33],      # Second Neuron
     [-0.44, 0.73, -0.13]]    # Third Neuron

b2 = [-1, 2, -0.5]


print()
layer1_outputs = np.dot(i, np.array(w).T) + b

layer2_outputs = np.dot(layer1_outputs, np.array(w2).T) + b2
print(layer2_outputs)


