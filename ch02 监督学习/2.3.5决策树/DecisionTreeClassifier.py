from mysklearn import *


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("DecisionTreeClassifier.png")
    plt.show()


if __name__ == '__main__':
    # mglearn.plots.plot_animal_tree()
    # plt.show()

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)
    print("X_train shape:{}".format(X_train.shape))  # X_train shape:(426, 30)
    print("y_train shape:{}".format(y_train.shape))  # y_train shape:(426,)

    print("X_test shape:{}".format(X_test.shape))  # X_test shape:(143, 30
    print("y_test shape:{}".format(y_test.shape))  # y_test shape:(143,)

    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)

    print("accuracy on train set:{:.3f}".format(tree.score(X_train, y_train)))  # accuracy on train set:1.000
    print("accuracy on test set:{:.3f}".format(tree.score(X_test, y_test)))  # accuracy on test set:0.930

    tree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree4.fit(X_train, y_train)
    print("accuracy on training set: {:.3f}".format(tree4.score(X_train, y_train)))
    print("accuracy on test set: {:.3f}".format(tree4.score(X_test, y_test)))

    export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names,
                    impurity=False, filled=True)

    # with open("tree.dot") as f:
    #     dot_graph = f.read()
    # graphviz.Source(dot_graph)

    print("feature importances:\n{}".format(tree.feature_importances_))
    # plot_feature_importances_cancer(tree4)

    tree = mglearn.plots.plot_tree_not_monotone()

    ram_prices = pd.read_csv("ram_price.csv")

    plt.semilogy(ram_prices.date, ram_prices.price)
    plt.xlabel("Year")
    plt.ylabel("Price in $/Mbyte")

    # 利用历史数据预测2000年后的价格
    data_train = ram_prices[ram_prices.date < 2000]
    data_test = ram_prices[ram_prices.date >= 2000]
    # 基于日期来预测价格
    X_train = data_train.date[:, np.newaxis]
    # 我们利用对数变换得到数据和目标之间更简单的关系
    y_train = np.log(data_train.price)
    tree_reg = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)
    # 对所有数据进行预测
    X_all = ram_prices.date[:, np.newaxis]
    pred_tree = tree_reg.predict(X_all)
    pred_lr = linear_reg.predict(X_all)
    # 对数变换逆运算
    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)
    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
    plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
    plt.legend()
    plt.savefig("ram_price.png")
    plt.show()
