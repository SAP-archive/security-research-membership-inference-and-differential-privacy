from abc import ABC, abstractmethod

import numpy as np


class DataSet(ABC):
    """
    Abstract base class that defines the interface for all data sets used in an MIApplication.
    """

    @abstractmethod
    def load_data(self):
        """
        Loads the training set and the test set. Training and test set must not be iterators.
        Instead they are supposed to be lists or numpy arrays
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        pass


class SingleClassUnbalancedDataSet(DataSet):
    """
    Abstract class that reduces the number of instances of one specific target class
    in order to create an unbalanced training data set w. r. t. that class.
    """

    def __init__(self, target_class, num_instances):
        """
        :param target_class: class to be reduced
        :param num_instances: final number of instances in the training set of target class or
                              fraction
        """
        super().__init__()
        self.target_class = target_class
        self.num_instances = num_instances

    def load_data(self):
        """
        based on the original data set that is supposed to be balanced, this method reduces the
        number of training instances by the desired amount provided in the constructor.
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        (x_train, y_train), (x_test, y_test) = self.load_balanced_data()

        mask = y_train == self.target_class

        if type(self.num_instances) == int:
            n = self.num_instances
        else:
            n = int(np.sum(mask) * float(self.num_instances))

        x_train_reduced = x_train[~mask]
        y_train_reduced = y_train[~mask]

        x_train = np.concatenate([x_train_reduced, x_train[mask][:n]])
        y_train = np.concatenate([y_train_reduced, y_train[mask][:n]])

        return (x_train, y_train), (x_test, y_test)

    @abstractmethod
    def load_balanced_data(self):
        """
        Loads the training set and the test set. Training and test set must not be iterators.
        Instead they are supposed to be lists or numpy arrays
        :return: two tuples --> (x_train, y_train), (x_test, y_test)
        """
        pass


class DataGenerator(ABC):
    """
    Every data generator passed to mia.core.model.Model must implement this interface
    """

    @abstractmethod
    def set_data(self, X, y):
        """
        Sets the data used for training and the batch size
        :param X_train:
        :param y_train:
        :return:
        """
        pass

