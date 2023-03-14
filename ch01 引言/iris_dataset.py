from myImport import *

if __name__ == '__main__':
    iris_dataset = load_iris()  # load_iris 返回的iris 对象是一个Bunch 对象，与字典非常相似，里面包含键和值：

    print("key of iris_dataset:\n{}".format(iris_dataset.keys()))
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    # print(iris_dataset['DESCR'][:193] + '\n...')
    print("target names:{}".format(iris_dataset['target_names']))
    # ['setosa' 'versicolor' 'virginica']
    print("feature names:{}".format(iris_dataset['feature_names']))
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # 花萼长度、花萼宽度、花瓣长度、花瓣宽度

    print("type of data:{}".format(type(iris_dataset['data'])))  # <class 'numpy.ndarray'>
    print("shape of data:{}".format(iris_dataset['data'].shape))  # (150, 4)
    # print("first five rows of data:\n{}".format(iris_dataset['data'][:5]))

    print("type of target:{}".format(type(iris_dataset['target'])))  # <class 'numpy.ndarray'>
    print("shape of target:{}".format(iris_dataset['target'].shape))  # (150,)
    # print("target:{}".format(iris_dataset['target']))

    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    print("X_train shape:{}".format(X_train.shape))  # (112, 4)
    print("y_train shape:{}".format(y_train.shape))  # (112,)
    print("X_test shape:{}".format(X_test.shape))  # (38, 4)
    print("y_test shape:{}".format(y_test.shape))  # (38,)

    # 利用iris_dataset.feature_names中的字符串对数据进行标记
    # iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20},cmap=mglearn.cm3)
    # plt.savefig("iris.png")
    # plt.show()


    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    X_new = np.array([[5, 2.9, 1, 0.2]])
    # print("X_new.shape{}".format(X_new.shape))
    prediction = knn.predict(X_new)
    print("prediction:{}".format(prediction))# [0]
    print("predicted target name:{}".format(iris_dataset['target_names'][prediction]))# ['setosa']

    y_pred = knn.predict((X_test))
    print("test set predictions:{}".format(y_pred))
    # print("test set score:{:.2f}".format(np.mean(y_pred == y_test)))
    print("test set score:{:.2f}".format(knn.score(X_test, y_test)))
