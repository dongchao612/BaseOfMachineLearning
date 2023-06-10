from mysklearn import *

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig("GradientBoostingClassifier.png")
    plt.show()


if __name__ == '__main__':
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)
    print("X_train shape:{}".format(X_train.shape))  # X_train shape:(426, 30)
    print("y_train shape:{}".format(y_train.shape))  # y_train shape:(426,)

    print("X_test shape:{}".format(X_test.shape))  # X_test shape:(143, 30
    print("y_test shape:{}".format(y_test.shape))  # y_test shape:(143,)

    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)

    print("accuracy on train set:{:.3f}".format(gbrt.score(X_train, y_train)))  # accuracy on train set:1.000
    print("accuracy on test set:{:.3f}".format(gbrt.score(X_test, y_test)))  # accuracy on test set:0.958

    gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt1.fit(X_train, y_train)

    print("accuracy on train set:{:.3f}".format(gbrt1.score(X_train, y_train)))  # accuracy on train set:0.988
    print("accuracy on test set:{:.3f}".format(gbrt1.score(X_test, y_test)))  # accuracy on test set:0.958

    gbrt001 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt001.fit(X_train, y_train)

    print("accuracy on train set:{:.3f}".format(gbrt001.score(X_train, y_train)))  # accuracy on train set:0.988
    print("accuracy on test set:{:.3f}".format(gbrt001.score(X_test, y_test)))  # accuracy on test set:0.937

    plot_feature_importances_cancer(gbrt)

    RandomForestClassifier()
    param_grid = {'n_estimators': [i for i in range(1, 200)]}
    print(param_grid)