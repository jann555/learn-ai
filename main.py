import numpy as np


def print_hi(name):
    arr = np.ndarray(shape=(1, 2), dtype=float, order='F')
    arr.put(0, 0.57)
    arr.put(1, 5.25)
    print(arr[0][1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
