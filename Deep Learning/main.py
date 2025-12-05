import numpy as np


def main():
    myMatrix = np.matrix([[0, 0, 0],
                          [1, 0, 1],
                          [0, 1, 1],
                          [1, 1, 1]])
    print(myMatrix)

    v = np.array([[2], [2], [2]])
    print(v)

def createVector(*args):
    
    for arg in args:
        if not isinstance(arg, int):
            raise("Error: args must be an integer")

    return np.array([[*args]])

if __name__ == "__main__":
# print(createVector(2, 3, 5, 5))
    # main()
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    unordered = np.array([3, 6, 8, 1, 2, 10])
    print(arr)
    print(arr.shape)
    print(f"Array Dimension: {arr.ndim}")
    print(f"Array Size: {arr.size}")
    print(f"Ordered Array: {np.sort(unordered)}")
