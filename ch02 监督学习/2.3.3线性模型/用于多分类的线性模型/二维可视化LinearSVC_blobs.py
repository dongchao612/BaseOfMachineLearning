from mglearn import discrete_scatter
from mglearn.plot_2d_separator import plot_2d_classification
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

from myImport import *

if __name__ == '__main__':
    # 加载数据
    X, y = make_blobs(random_state=42)

    # 构建、训练模型
    linear_svm = LinearSVC().fit(X, y)

    plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
    discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                      ['b', 'r', 'g']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
                'Line class 2'], loc=(1.01, 0.3))
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("二维可视化LinearSVC_blobs.png")
    plt.show()