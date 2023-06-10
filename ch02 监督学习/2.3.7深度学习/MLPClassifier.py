from mysklearn import *

if __name__ == '__main__':
    # mglearn.plots.plot_logistic_regression_graph()
    # mglearn.plots.plot_single_hidden_layer_graph()
    # mglearn.plots.plot_two_hidden_layer_graph()
    # plt.show()

    # line = np.linspace(-3, 3, 100)
    # plt.plot(line, np.tanh(line), label="tanh")
    # plt.plot(line, np.maximum(line, 0), label="relu")
    # plt.legend(loc="best")
    # plt.xlabel("x")
    # plt.ylabel("relu(x), tanh(x)")
    # plt.show()

    # X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    # mlp = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10])
    # mlp10 = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10])
    # mlp1010 = MLPClassifier(solver='lbfgs', random_state=0,activation="tanh",hidden_layer_sizes=[10,10])
    # mlp1010.fit(X_train, y_train)
    # mglearn.plots.plot_2d_separator(mlp1010, X_train, fill=True, alpha=.3)
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")

    # fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    # for axx, n_hidden_nodes in zip(axes, [10, 100]):
    #     for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
    #         mlp = MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],alpha=alpha)
    #     mlp.fit(X_train, y_train)
    #     mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    #     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    #     ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
    #         n_hidden_nodes, n_hidden_nodes, alpha))
    # plt.show()

    # fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    # for i, ax in enumerate(axes.ravel()):
    #     mlp = MLPClassifier(solver='lbfgs', random_state=i,hidden_layer_sizes=[100, 100])
    #     mlp.fit(X_train, y_train)
    #     mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
    #     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    # plt.show()

    cancer = load_breast_cancer()

    print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    print("accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
    print("accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

    # 计算训练集中每个特征的平均值
    mean_on_train = X_train.mean(axis=0)
    # 计算训练集中每个特征的标准差
    std_on_train = X_train.std(axis=0)
    # 减去平均值，然后乘以标准差的倒数
    # 如此运算之后，mean=0，std=1
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    # 对测试集做相同的变换（使用训练集的平均值和标准差）
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    mlp = MLPClassifier(random_state=0, max_iter=1000)
    mlp1 = MLPClassifier(random_state=0, alpha=1, max_iter=1000)
    mlp1.fit(X_train_scaled, y_train)
    print("accuracy on training set: {:.3f}".format(mlp1.score(X_train_scaled, y_train)))
    print("accuracy on test set: {:.3f}".format(mlp1.score(X_test_scaled, y_test)))

    plt.figure(figsize=(20, 5))
    plt.imshow(mlp1.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()
    plt.show()
