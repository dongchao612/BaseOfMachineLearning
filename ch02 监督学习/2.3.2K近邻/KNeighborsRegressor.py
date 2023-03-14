from myImport import *

if __name__ == '__main__':
    plot_knn_regression(n_neighbors=3)
    plt.savefig("plot_knn_regression_3.png")
    plt.show()

    X, y = make_wave()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print("X_train shape:{}".format(X_train.shape))  # X_train shape:(75, 1)
    # print("y_train shape:{}".format(y_train.shape))  # y_train shape:(75,)
    # print("X_test shape:{}".format(X_test.shape))  # X_test shape:(25, 1)
    # print("y_test shape:{}".format(y_test.shape))  # y_test shape:(25,)

    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)

    print("test set predictions: \n{}".format(reg.predict(X_test)))
    print("test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

    # 分析KNeighborsRegressor
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # 创建1000个数据点，在-3和3之间均匀分布
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # 利用1个、3个或9个邻居分别进行预测
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
        ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                n_neighbors, reg.score(X_train, y_train),
                reg.score(X_test, y_test)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target",
                    "Test data/target"], loc="best")
    plt.savefig("KNeighborsRegressor.png")
    plt.show()
