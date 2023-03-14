from myImport import *

if __name__ == '__main__':
    X, y = make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(['Class 0', 'Class 1'], loc=4)
    plt.savefig("forge.png")
    plt.show()

    print("X.shape{}".format(X.shape))  # X.shape(26, 2)
    print({n: v for n, v in zip(['Class 0', 'Class 1'], np.bincount(y))})  # {'Class 0': 13, 'Class 1': 13}
