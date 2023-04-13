from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from myImport import *

if __name__ == '__main__':
    # 加载数据集
    iris_dataset = load_iris()  # load_iris 返回的iris 对象是一个Bunch 对象，与字典非常相似，里面包含键和值：

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    # 利用iris_dataset.feature_names中的字符串对数据进行标记
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # print("iris_dataframe:\n...",iris_dataframe)
    scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20},
                               cmap=mglearn.cm3)
    plt.savefig("iris_dataset.png")
    plt.show()