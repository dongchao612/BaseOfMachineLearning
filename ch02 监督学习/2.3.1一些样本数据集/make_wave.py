import matplotlib.pyplot as plt
from mglearn.datasets import make_wave

if __name__ == '__main__':
    X, y = make_wave(100)  # X.shape(100, 1)
    plt.plot(X, y, "o")
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.savefig("wave.png")
    plt.show()
