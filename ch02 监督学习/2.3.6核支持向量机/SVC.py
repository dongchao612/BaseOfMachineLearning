from mysklearn import *

if __name__ == '__main__':
    # X, y = mglearn.tools.make_handcrafted_dataset()
    # svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    # mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # # 画出支持向量
    # sv = svm.support_vectors_
    # # 支持向量的类别标签由dual_coef_的正负号给出
    # sv_labels = svm.dual_coef_.ravel() > 0
    # mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")
    # plt.savefig("SVC.png")
    # plt.show()

    # fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    # for ax, C in zip(axes, [-1, 0, 3]):
    #     for a, gamma in zip(ax, range(-1, 2)):
    #         mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    # axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
    #                   ncol=4, loc=(.9, 1.2))
    # plt.show()

    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    svc = SVC()
    svc.fit(X_train, y_train)
    print("accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
    print("accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

    # plt.plot(X_train.min(axis=0), 'o', label="min")
    # plt.plot(X_train.max(axis=0), '^', label="max")
    # plt.legend(loc=4)
    # plt.xlabel("Feature index")
    # plt.ylabel("Feature magnitude")
    # plt.yscale("log")
    # plt.show()

    # 计算训练集中每个特征的最小值
    min_on_training = X_train.min(axis=0)
    # 计算训练集中每个特征的范围（最大值-最小值）
    range_on_training = (X_train - min_on_training).max(axis=0)
    # 减去最小值，然后除以范围
    # 这样每个特征都是min=0和max=1
    X_train_scaled = (X_train - min_on_training) / range_on_training
    print("minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
    print("maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

    # 利用训练集的最小值和范围对测试集做相同的变换（详见第3章）
    X_test_scaled = (X_test - min_on_training) / range_on_training

    svc = SVC()
    svc.fit(X_train_scaled, y_train)
    print("accuracy on training set: {:.3f}".format(
        svc.score(X_train_scaled, y_train)))
    print("accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
