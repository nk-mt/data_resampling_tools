from collections import Counter
from random import sample, shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.datasets import fetch_datasets
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors


def iris_dataset():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()
    scat = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg',
                       edgecolor='k', marker="o")
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    h1, l1 = scat.legend_elements()
    plt.legend(h1, ["Iris-setosa"] + ['Iris-versicolor'] + ["Iris-virginica"])
    plt.show()


def iris_pair_plot():
    iris = sns.load_dataset("iris")
    sns.pairplot(iris, hue="species")
    plt.show()


def make_plot_despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([0., 3.5])
    ax.set_ylim([0., 3.5])
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.legend()
    plt.show()


def nm1():

    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])

    X_majority = np.transpose([[2.1, 3.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45],
                               [1.5, 1.7, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9]])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative class', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Positive class', s=200, marker='+')

    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Avg distance={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    dist, ind = nearest_neighbors.kneighbors(X_majority)
    av_dist_index = np.argsort([sum(d)/3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[:7]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative samples selected by NM1', s=200, alpha=0.3, color='g')

    ax.set_title('NearMiss-1')
    make_plot_despine(ax)


def nm2():

    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])

    X_majority = np.transpose([[2.1, 2.13, 2.12, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5, 1.5],
                               [1.5, 2.7, 2.1, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3, 2.2]])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative class', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Positive class', s=200, marker='+')

    nearest_neighbors = NearestNeighbors(n_neighbors=X_minority.shape[0])
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist = dist[:, -3::]
    ind = ind[:, -3::]
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Avg distance={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    ax.set_title('NearMiss-2')

    dist, ind = nearest_neighbors.kneighbors(X_majority)
    dist = dist[:, -3::]
    av_dist_index = np.argsort([sum(d) / 3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[:7]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative samples selected by NM2', s=200, alpha=0.3, color='g')

    make_plot_despine(ax)


def nm3():

    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])

    X_majority = np.transpose([[2.1, 1.5, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5],
                               [1.5, 2.2, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3]])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative class', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Positive class', s=200, marker='+')

    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_majority)


    selected_idx = nearest_neighbors.kneighbors(X_minority, return_distance=False)
    X_majority = X_majority[np.unique(selected_idx), :]
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='М nearest nighbours to positive samples', s=800, alpha=0.5, color='y')
    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Avg distance={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    ax.set_title('NearMiss-3')
    dist, ind = nearest_neighbors.kneighbors(X_majority)
    av_dist_index = np.argsort([sum(d) / 3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[-7:]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative samples selected by NM3', s=200, alpha=0.3, color='g')
    make_plot_despine(ax)

    fig.tight_layout()


def ru():


    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])

    X_majority = np.transpose([[2.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5, 1.5],
                               [1.5, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3, 2.2]])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Negative class', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Positive class', s=200, marker='+')

    selected_idx = take_unique_indexes(9, 7)

    X_majority = X_majority[np.unique(selected_idx), :]
    ax.scatter(X_majority[:, 0], X_majority[:, 1], label='Randomly selected negative samples for removal', s=200, alpha=0.3, color='g')

    ax.set_title('Random Under-sampling')
    make_plot_despine(ax)
    fig.tight_layout()


def take_unique_indexes(high_index, amount):
    indexes = np.unique([])
    while len(indexes) != amount:
        indexes = np.unique(np.random.randint(0, high_index, amount))
    return indexes

def load_ds(path):
    df = pd.read_csv(path, sep=',', header=None)
    values = df.values
    return values


def print_examples():

    ts = fetch_datasets()['thyroid_sick']
    print(ts.data.shape)

    print(sorted(Counter(ts.target).items()))
    ds = load_ds('thyroid_sick.data')
    labels = ['Target classes']
    healty, sick = ([len(list(filter(lambda x: x[-1] == 0, ds)))], [len(list(filter(lambda x: x[-1] == 1, ds)))])


    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, healty, width, label='Healthy')
    rects2 = ax.bar(x + width / 2, sick, width, label='Sick')

    ax.set_ylabel('Number of samples')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()


def load_transformed_ds(path, subset, negative_samples, positive_samples):
    df = load_ds(path)
    if subset:
        Xy_sub_0 = sample(list(filter(lambda x: x[-1] == 0, df)), negative_samples)
        Xy_sub_1 = sample(list(filter(lambda x: x[-1] == 1, df)), positive_samples)
        df = Xy_sub_0 + Xy_sub_1
        shuffle(df)
    X, y = (list(map(lambda x: x[:-1], df)), list(map(lambda x: x[-1], df)))
    return (np.array(X), np.array(y));


def logistic_regression():
    X, y = make_classification(1000, 2, 2, 0, weights=[.99, .01], random_state=15)
    X_test, y_test = make_classification(1000, 2, 2, 0, weights=[.99, .01], random_state=15)
    clf = LogisticRegression().fit(X, y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    predicted_y_s = clf.predict(X_test)
    print (accuracy_score(y_test, predicted_y_s))
    cm = confusion_matrix(y_test, predicted_y_s)

    probs = clf.predict_proba(grid)[:, 0].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))
    cont = ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    scatt1 = ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c=np.array(list(filter(lambda x : x == 0, y_test))), s=100,
               cmap="RdBu", vmin=-.2, vmax=1.2, marker="x", label="0",
               edgecolor="white", linewidth=1)
    scatt2 = ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c=np.array(list(filter(lambda x: x == 1, y_test))),
               s=100,
               cmap="RdBu", vmin=-.2, vmax=1.2, marker="o", label="1",
               edgecolor="white", linewidth=1)

    h1,l1 = cont.legend_elements()
    h2,l2 = scatt1.legend_elements()
    h3,l3 = scatt2.legend_elements()


    ax.legend(h1 + h2 + h3, ["P(y=1)"] + ["y=0"]  + ["y=1"])

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return cm


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 2))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'size': 10})


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.yticks(tick_marks, target_names, fontsize=10)
        plt.xticks(tick_marks, target_names, fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center", fontdict={'size': 10},
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center", fontdict={'size': 10},
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('Actual class', fontdict={'size': 10})
    plt.xlabel('Predicted class\n\nAccuracy={:0.4f}'.format(accuracy), fontdict={'size': 10})
    plt.show()

if __name__ == '__main__':
    iris_dataset()
    iris_pair_plot()
    nm1()
    nm2()
    nm3()
    ru()
    print_examples()
    cm = logistic_regression()
    plot_confusion_matrix(cm,
                          normalize=False,
                          target_names=['0', '1'],
                          title="Confusion Matrix")