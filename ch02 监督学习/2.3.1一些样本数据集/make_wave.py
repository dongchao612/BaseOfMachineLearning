from myImport import *

if __name__ == '__main__':
    X, y = make_wave()  # X.shape(100, 1)
    plt.plot(X, y, "o")
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.savefig("wave.png")
    plt.show()
