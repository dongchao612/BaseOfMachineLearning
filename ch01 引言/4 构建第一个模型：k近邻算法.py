from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from myImport import *

if __name__ == '__main__':
    # 加载数据集
    iris_dataset = load_iris()  # load_iris 返回的iris 对象是一个Bunch 对象，与字典非常相似，里面包含键和值：

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    # 构建第一个模型
    knn = KNeighborsClassifier(n_neighbors=1)

    # 训练
    knn.fit(X_train, y_train)

    X_new = np.array([[5, 2.9, 1, 0.2]])
    print("X_new.shape{}".format(X_new.shape))

    # 做出预测
    prediction = knn.predict(X_new)
    print("prediction:{}".format(prediction))  # [0]
    print("predicted target name:{}".format(iris_dataset['target_names'][prediction]))  # ['setosa']

    # 评估模型
    y_pred = knn.predict((X_test))
    print("test set predictions:{}".format(y_pred))
    print("test set score:{:.2f}".format(np.mean(y_pred == y_test)))



