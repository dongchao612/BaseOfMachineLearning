from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from myImport import *
if __name__ == '__main__':
    # 加载数据
    cancer = load_breast_cancer()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)

    logreg = LogisticRegression().fit(X_train, y_train)
    print("logreg train set score: {:.3f}".format(logreg.score(X_train, y_train)))  # 0.953
    print("logreg test set score: {:.3f}".format(logreg.score(X_test, y_test)))  # 0.965

    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print("logreg100 train set score: {:.3f}".format(logreg100.score(X_train, y_train)))  # 0.946
    print("logreg100 test set score: {:.3f}".format(logreg100.score(X_test, y_test)))  # 0.958

    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print("logreg001 Train set score: {:.3f}".format(logreg001.score(X_train, y_train)))  # 0.937
    print("logreg001 Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))  # 0.930

    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.savefig("LogisticRegression_cancer.png")
    plt.show()

