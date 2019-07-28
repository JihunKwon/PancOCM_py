import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def main():
    iris = load_iris()
    pca = PCA(n_components=2)
    data = pca.fit_transform(iris.data)

    x = np.linspace(-5, 5, 500)
    y = np.linspace(-1.5, 1.5, 250)
    X, Y = np.meshgrid(x, y)

    ocsvm = OneClassSVM(nu=0.1, gamma="auto")
    ocsvm.fit(data)
    df = ocsvm.decision_function(
        np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    preds = ocsvm.predict(data)

    plt.scatter(data[:,0], data[:,1], c=preds,
                cmap=plt.cm.RdBu, alpha=0.8)
    r = max([abs(df.min()), abs(df.max())])
    plt.contourf(X, Y, df, 10, vmin=-r, vmax=r,
                 cmap=plt.cm.RdBu, alpha=.5)
    plt.savefig("result.png")

if __name__ == "__main__":
    main()
