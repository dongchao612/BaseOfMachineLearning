from mglearn.datasets import load_extended_boston
from sklearn.datasets import load_boston

if __name__ == '__main__':
    boston = load_boston()
    print("data shape: {}".format(boston.data.shape))  # data shape: (506, 13)

    X, y = load_extended_boston()  # 最初的13个特征加上这13个特征两两组合（有放回）得到的91个特征，一共有104个特征
    print("X.shape: {}".format(X.shape))  # X.shape: (506, 104)
