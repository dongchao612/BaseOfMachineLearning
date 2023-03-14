from mysklearn import *

if __name__ == '__main__':
    X, y = make_blobs(random_state=42)
    # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.legend(["Class 0", "Class 1", "Class 2"])
    # plt.show()

    linear_svm = LinearSVC()
    linear_svm.fit(X, y)
    print("Coefficient shape: ", linear_svm.coef_.shape)  # Coefficient shape:  (3, 2)
    print("Intercept shape: ", linear_svm.intercept_.shape)  # Intercept shape:  (3,)

    mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.savefig("LinearSVC.png")
    plt.show()
