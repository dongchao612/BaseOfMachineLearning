
from mysklearn import  *
if __name__ == '__main__':
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.show()

    linear_svm = LinearSVC().fit(X, y)
    #mglearn.plots.plot_2d_separator(linear_svm, X)
    # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.savefig("LinearSVC.png")
    # plt.show()

    # 添加第二个特征的平方，作为一个新特征
    # X_new = np.hstack([X, X[:, 1:] ** 2])
    # figure = plt.figure()
    # 3D可视化
    #ax = Axes3D(figure, elev=-152, azim=-26)
    # 首先画出所有y == 0的点，然后画出所有y == 1的点
    # mask = y == 0
    # ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
    #            cmap=mglearn.cm2, s=60)
    # ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
    #            cmap=mglearn.cm2, s=60)
    # ax.set_xlabel("feature0")
    # ax.set_ylabel("feature1")
    # ax.set_zlabel("feature1 ** 2")
    # plt.savefig("LinearSVC_Axes3D.png")
    # plt.show()

    X_new = np.hstack([X, X[:, 1:] ** 2])
    linear_svm_3d = LinearSVC()
    linear_svm_3d.fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # 显示线性决策边界
    # figure = plt.figure()
    # ax = Axes3D(figure, elev=-152, azim=-26)
    # mask = y == 0
    # xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    # yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    # XX, YY = np.meshgrid(xx, yy)
    # ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
    # ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    # ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
    #            cmap=mglearn.cm2, s=60)
    # ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
    #            cmap=mglearn.cm2, s=60)
    # ax.set_xlabel("feature0")
    # ax.set_ylabel("feature1")
    # ax.set_zlabel("feature1 ** 2")
    # plt.show()

    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = YY ** 2
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
                 cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


