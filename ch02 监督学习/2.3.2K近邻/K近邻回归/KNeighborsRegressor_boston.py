from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from myImport import *

if __name__ == '__main__':
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=66)

    training_accuracy = []  # 训练集精度
    test_accuracy = []  # 泛化精度

    # n_neighbors取值从1到10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # 构建模型
        clf = KNeighborsRegressor(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)

        # 记录训练集精度
        training_accuracy.append(clf.score(X_train, y_train))
        # 记录泛化精度
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig("KNeighborsRegressor_boston.png")
    plt.show()
