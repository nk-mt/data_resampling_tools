import numpy as np
from enum import Enum
from sklearn import svm, tree


class ClassificationAlgorithms(Enum):

    CART = ("CART (Decision Tree Classifier)", tree.DecisionTreeClassifier(criterion='entropy', random_state=np.random.RandomState(1)))
    SVM = ("SVM", svm.SVC(probability=True, random_state=np.random.RandomState(1)))

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ca: ca.value[0] == name, ClassificationAlgorithms)
        return next(filtered_algos, ClassificationAlgorithms.CART)