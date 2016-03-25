#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.array([0, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6])
    y = np.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4])
    x = x / 6.
    y = y / 4.
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
