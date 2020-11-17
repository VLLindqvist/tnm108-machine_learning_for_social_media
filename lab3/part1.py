from sklearn import datasets
from sklearn.model_selection import train_test_split
from GaussNB import GaussNB, np


def main():
    nb = GaussNB()
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    nb.target_values = np.unique(target)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.3)
    nb.train(X_train, y_train)
    predicted = nb.predict(X_test)
    accuracy = nb.accuracy(y_test, predicted)
    print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    main()
