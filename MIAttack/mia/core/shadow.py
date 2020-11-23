import gc
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class ShadowGroup:
    """
    A group of shadow models that can be trained uniformly. The group creates individual training
    data sets for each model and tracks which model used which sample for training. This feature
    will ease the construction of the attack data set later on.
    """

    def __init__(
        self,
        num_shadows,
        shadow_model_factory,
        X_prior=None,
        y_prior=None,
        memory_efficiency=False,
        path_template=None,
    ):
        """
        Creates a ShadowGroup which consists of a number of shadow models each being created by
        the shadow_model_factory. shadow_model_factory is supposed to create models fulfilling
        the Model interface in model.py.
        :param num_shadows: number of shadow models
        :param shadow_model_factory: model to generate shadow models
        :param memory_efficiency: use only one physical model to represent the shadow group
        :param path_template: path to save models to is model_efficiency true
        """
        self.memory_efficiency = memory_efficiency
        if self.memory_efficiency:
            self.models = None
        else:
            self.models = [shadow_model_factory.create() for _ in range(num_shadows)]
        self.indices = []
        self.X_train = None
        self.y_train = None
        self.X_train_attacker = None
        self.y_train_attacker = None
        self.X_prior = X_prior
        self.y_prior = y_prior
        self.num_shadows = num_shadows
        self.model_factory = shadow_model_factory
        if path_template:
            self.path_template = path_template
        else:
            self.path_template = os.path.join("/", "tmp", "mia_shadow_model_{}.h5")

    def fit(self, X_shadow, y_shadow, num_samples):
        """
        Fits the each shadow model with a subset of X_shadow. That subset of X_shadow contains
        num_samples elements. ShadowGroup tracks the indices of the train subset and of the test
        subset and stores them in a list that can be retrieved via
        ShadowGroup.get_train_test_indices()
        :param X_shadow: list of data samples
        :param y_shadow: list of labels
        :param num_samples: number of samples to train each shadow model (also the size of the test
        set)
        """

        if num_samples >= len(X_shadow) / 2:
            raise ValueError(
                "num_samples must be smaller than half of the training set size"
            )

        self.indices = []
        self.X_train = np.array(X_shadow)
        self.y_train = np.array(y_shadow)

        train_samples = (
            num_samples if self.X_prior is None else num_samples - len(self.X_prior)
        )
        test_samples = num_samples

        shadow_history = []

        for i in range(self.num_shadows):
            # generate a balanced subset for training and testing
            (
                train_indices,
                test_indicces,
            ) = ShadowGroup._generate_shadow_train_test_indices(
                self.X_train, self.y_train, train_samples, test_samples
            )

            X_shadow_train = self.X_train[train_indices]
            y_shadow_train = self.y_train[train_indices]
            X_shadow_test = self.X_train[test_indicces]
            y_shadow_test = self.y_train[test_indicces]

            if self.X_prior is not None and self.y_prior is not None:
                X_shadow_train = np.concatenate([X_shadow_train, self.X_prior])
                y_shadow_train = np.concatenate([y_shadow_train, self.y_prior])

            if self.memory_efficiency:
                model = self.model_factory.create()
                single_shadow_history = model.fit(
                    X_train=X_shadow_train,
                    y_train=y_shadow_train,
                    X_validation=X_shadow_test,
                    y_validation=y_shadow_test,
                )
                model.save(self.path_template.format(i))
                model.clear_memory()
                del model
                gc.collect()
            else:
                single_shadow_history = self.models[i].fit(
                    X_train=X_shadow_train,
                    y_train=y_shadow_train,
                    X_validation=X_shadow_test,
                    y_validation=y_shadow_test,
                )
            self.indices.append((train_indices, test_indicces))
            shadow_history.append(single_shadow_history.history)

        return shadow_history

    def get_models(self):
        """
        Returns a generator to loop over all shadow models. If memory_efficiency flag is set,
        the caller needs to take care of proper clean up.
        :return: list of shadow models
        """
        for i in range(self.num_shadows):
            if self.memory_efficiency:
                model = self.model_factory.create()
                model.load(self.path_template.format(i))
            else:
                model = self.models[i]
            yield model
            del model
            gc.collect()

    def size(self):
        """
        :return: number of shadow models contained in the group
        """
        return self.num_shadows

    def get_prior_knowledge(self):
        return self.X_prior, self.y_prior

    def get_train_test_indices(self):
        """
        Provides a list of indices that were used to train each shadow model,
        i. e. the indices of a subset of samples passed to ShadowGroup.fit as X_train, as well as
        the test indices. The order of the indices list corresponds to the order of the models list.
        Each entry in the list is a tuple containing train indices and test indices in that order.
        :return: list of indices as numpy array
        """
        return np.array(self.indices)

    def get_group_data_set(self):
        """
        Returns the data set used to train and test all shadow models. The indices returned by
        get_train_test_indices refer to this set
        :return: X_train, y_train as numpy arrays
        """

        return self.X_train, self.y_train

    def set_attack_data_set(self, X_train_attacker, y_train_attacker):
        """
        Sets the dataset used to train the attack model.
        This is useful for the ldp case, where the attacker can decide to attack in the ldp or the orig domain.
        :param X_shadow: list of data samples
        :param y_shadow: list of labels
        """
        self.X_train_attacker = np.array(X_train_attacker)
        self.y_train_attacker = np.array(y_train_attacker)

    def get_attack_data_set(self):
        """
        Returns the data set specified for attacker training. 
        This is similar to get_group_data_set() if no other data set was specified with set_attack_data_set().
        :return: X_train, y_train as numpy arrays
        """

        return self.X_train_attacker, self.y_train_attacker

    @staticmethod
    def _generate_shadow_train_test_indices(
        X_train, y_train, train_samples, test_samples
    ):
        """
        Internal function to create equally sized and distributed train and test sets for each
        shadow model
        :param samples_per_shadow: samples in each training and test set
        :return: the train and test indices of one pair of train and test sets
        """
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_samples, test_size=test_samples
        )

        shadow_train_indices, shadow_test_indices = list(sss.split(X_train, y_train))[0]

        return shadow_train_indices, shadow_test_indices
