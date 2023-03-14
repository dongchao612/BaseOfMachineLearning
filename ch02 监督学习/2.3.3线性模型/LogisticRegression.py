from mysklearn import *

if __name__ == '__main__':
    # X, y = make_forge()
    # fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    #     clf = model.fit(X, y)
    #     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    #                                     ax=ax, alpha=.7)
    #     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    #     ax.set_title("{}".format(clf.__class__.__name__))
    #     ax.set_xlabel("Feature 0")
    #     ax.set_ylabel("Feature 1")
    # axes[0].legend()
    # plt.savefig("LinearSVC_LogisticRegression.png")
    # plt.show()
    #
    # mglearn.plots.plot_linear_svc_regularization()
    # plt.show()


    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print("logreg train set score: {:.3f}".format(logreg.score(X_train, y_train)))
    print("logreg test set score: {:.3f}".format(logreg.score(X_test, y_test)))

    logreg100 = LogisticRegression(C=100)
    logreg100.fit(X_train, y_train)
    print("logreg100 train set score: {:.3f}".format(logreg100.score(X_train, y_train)))
    print("logreg100 test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

    logreg001 = LogisticRegression(C=0.01)
    logreg001.fit(X_train, y_train)
    print("logreg001 Train set score: {:.3f}".format(logreg001.score(X_train, y_train)))
    print("logreg001 Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

    # plt.plot(logreg.coef_.T, 'o', label="C=1")
    # plt.plot(logreg100.coef_.T, '^', label="C=100")
    # plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    # plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    # plt.hlines(0, 0, cancer.data.shape[1])
    # plt.ylim(-5, 5)
    # plt.xlabel("Coefficient index")
    # plt.ylabel("Coefficient magnitude")
    # plt.legend()
    # plt.savefig("logreg.png")
    # plt.show()
    #
    # for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    #     lr_l1 = LogisticRegression(C=C, penalty="l1",solver='liblinear')
    #     lr_l1.fit(X_train, y_train)
    #     print("Training accuracy of l2 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
    #     print("Test accuracy of l2 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
    # plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    # plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    # plt.hlines(0, 0, cancer.data.shape[1])
    # plt.xlabel("Coefficient index")
    # plt.ylabel("Coefficient magnitude")
    # plt.ylim(-5, 5)
    # plt.legend(loc=3)
    # plt.savefig("l1.png")
    # plt.show()
