from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 加载数据集
    iris_dataset = load_iris()  # load_iris 返回的iris 对象是一个Bunch 对象，与字典非常相似，里面包含键和值：

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    print("X_train shape:{}".format(X_train.shape))  # (112, 4)
    print("y_train shape:{}".format(y_train.shape))  # (112,)
    print("X_test shape:{}".format(X_test.shape))  # (38, 4)
    print("y_test shape:{}".format(y_test.shape))  # (38,)
