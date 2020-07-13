from enum import Enum

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, CondensedNearestNeighbour, OneSidedSelection, \
    NearMiss, EditedNearestNeighbours, NeighbourhoodCleaningRule, InstanceHardnessThreshold, \
    RepeatedEditedNearestNeighbours, AllKNN

from algs import smote_boost


class ResamplingAlgorithms(Enum):
    RO = ("Random Over-sampling", RandomOverSampler(random_state=1))
    SMOTE = ("Smote", SMOTE(random_state=1))
    ADASYN = ("ADASYN", ADASYN(random_state=1))
    SMOTE_TL = ('SMOTE+TL', SMOTETomek(random_state=1))
    SMOTE_ENN = ('SMOTE+ENN', SMOTEENN(random_state=1))
    SMOTE_BOOST = ("SMOTEBoost", smote_boost.SMOTEBoost(random_state=1))
    RU = ("Random Under-sampling", RandomUnderSampler(random_state=1))
    CLUSTERCENTROIDS = ("ClusterCentroids", ClusterCentroids(random_state=1))
    TOMEK_LINKS = ("TomekLinks", TomekLinks())
    NM1 = ("NM1", NearMiss(version=1))
    NM2 = ("NM2", NearMiss(version=2))
    NM3 = ("NM3", NearMiss(version=3))
    CNN = ("CNN", CondensedNearestNeighbour(random_state=1))
    OSS = ("OneSidedSelection", OneSidedSelection(random_state=1))
    ENN = ('ENN', EditedNearestNeighbours())
    NCL = ('NCL', NeighbourhoodCleaningRule())
    IHT = ('IHT', (InstanceHardnessThreshold(random_state=1)))
    RENN = ('RENN', RepeatedEditedNearestNeighbours())
    AllKNN = ('AllKNN', AllKNN())

    @classmethod
    def get_algorithm_by_name(cls, name):
        filtered_algos = filter(lambda ra: ra.value[0] == name, ResamplingAlgorithms)
        return next(filtered_algos, ResamplingAlgorithms.RO)