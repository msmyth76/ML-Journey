import math
import numpy as np 

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]



# Exponentiation + Normalization = softmax
exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# for o in layer_outputs:
#     exp_values.append(E**o)

# norm_base = sum(exp_values)
# norm_values = []
#
# for v in exp_values:
#     norm_values.append(v / norm_base)

print(exp_values)
print(norm_values)
print(sum(norm_values))
