from sklearn.datasets import load_iris

if __name__ == '__main__':
    # 加载数据集
    iris_dataset = load_iris()  # load_iris 返回的iris 对象是一个Bunch 对象，与字典非常相似，里面包含键和值：

    print("key of iris_dataset:\n{}".format(iris_dataset.keys()))
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    print(iris_dataset['DESCR'][:193] + '\n...')

    print("target names:{}".format(iris_dataset['target_names']))
    # ['setosa' 'versicolor' 'virginica']

    print("feature names:{}".format(iris_dataset['feature_names']))
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 花萼长度、花萼宽度、花瓣长度、花瓣宽度

    print("type of data:{}".format(type(iris_dataset['data'])))  # <class 'numpy.ndarray'>
    print("shape of data:{}".format(iris_dataset['data'].shape))  # (150, 4)
    print("first five rows of data:\n{}".format(iris_dataset['data'][:5]))

    print("type of target:{}".format(type(iris_dataset['target'])))  # <class 'numpy.ndarray'>
    print("shape of target:{}".format(iris_dataset['target'].shape))  # (150,)
    print("first five rows of  target:{}".format(iris_dataset['target'][:5]))

