from myImport import *

if __name__ == '__main__':
    cancer = load_breast_cancer()
    print("key of cancer:\n{}".format(cancer.keys()))
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    print("shape of data:{}".format(cancer.data.shape))  # shape of data:(569, 30)
    print("sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))  # {'malignant': 212, 'benign': 357}
    print("feature names:\n{}".format(cancer.feature_names))  # 30个
