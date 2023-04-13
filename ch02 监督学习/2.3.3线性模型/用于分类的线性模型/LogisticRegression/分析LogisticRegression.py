from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from  myImport import *
if __name__ == '__main__':
    # 加载数据
    cancer = load_breast_cancer()

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)
    # 分析LogisticRegression
    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C, penalty="l2").fit(X_train, y_train)
        print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
        print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
        plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    plt.savefig("分析LogisticRegression.png")
    plt.show()
