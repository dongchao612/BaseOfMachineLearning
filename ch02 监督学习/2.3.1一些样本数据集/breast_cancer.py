from pandas.plotting import scatter_matrix
from sklearn.datasets import load_breast_cancer

from myImport import *

if __name__ == '__main__':
    # 加载数据集
    cancer = load_breast_cancer()

    print("key of cancer:\n{}".format(cancer.keys()))
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

    print("shape of data:{}".format(cancer.data.shape))  # shape of data:(569, 30)

    print(
        "sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
    # {'malignant': 212, 'benign': 357}

    print("feature names:\n{}".format(cancer.feature_names))
    # 特征名字
    ''' 
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    '''

    _dataframe = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    scatter_matrix(_dataframe, c=cancer.target, figsize=(150, 150), marker='o', hist_kwds={'bins': 20},
                   cmap=mglearn.cm3)
    plt.savefig("cancer_matrix.png")
    plt.show()
