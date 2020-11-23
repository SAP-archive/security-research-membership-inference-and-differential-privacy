import gc
import os

import numpy as np
from sklearn.utils import shuffle


class Attacker:
    """
    Class that contains a list of attack models for several target classes.
    Before attacking a class, the attacker must be trained for it. Afterwards, attacks against
    that class can be performed
    """

    def __init__(self, attack_model_factory, target_classes, memory_efficiency=False):
        """
        Creates an attacker object managing multiple attack models. Each attack
        model is created by using a model factory
        :param attack_model_factory: model factory that generates attack models
        """
        self.__model_factory = attack_model_factory
        self.memory_efficiency = memory_efficiency
        self.attack_models = {}
        self.path_template = os.path.join("/", "tmp", "mia_attack_model_{}.h5")
        self.target_classes = target_classes

    def train_attack_model(
        self, target_class, X_train, y_train, X_val=None, y_val=None
    ):
        """
        Trains the attacker to infer membership for a specific target_class.
        :param X_train: training set for that class
        :param y_train: labels of 0 and 1
        :param target_class: class to attack with that model
        """
        model = self.__model_factory.create()
        single_attack_history = model.fit(X_train, y_train, X_val, y_val)
        if self.memory_efficiency:
            model.save(self.path_template.format(target_class))
        else:
            self.attack_models[target_class] = model

        return single_attack_history

    def attack_target_class(self, X_test, target_class):
        """
        Attacks outputs X_train of a target model w.r.t. a specific target class
        :param X_test: outputs of target model
        :param target_class: class to attack
        :return: membership inference prediction
        """

        model = self.get_model(target_class)
        predictions = model.predict(X_test)

        if self.memory_efficiency:
            model.clear_memory()
            del model
            gc.collect()

        return predictions

    def evaluate_attack(self, X_test, y_test, target_class, metrics):
        """
        Evaluates the attack against a specific target class using metrics
        :param X_test: outputs of the target model
        :param y_test: true membership labels
        :param target_class: class to attack
        :param metrics: metrics to evaluate, e. g. precision or recall
        :return: metric score for the attack and confidence values
        """
        y_pred = self.attack_target_class(X_test, target_class)
        y_hot_pred = np.argmax(y_pred, axis=1)
        results = [metric(y_test, y_hot_pred) for metric in metrics]
        return results, y_pred

    def get_models(self):
        """
        Generator over all attack models.
        :return: all attack models
        """
        for target_class in self.target_classes:
            model = self.get_model(target_class)
            yield model
            del model
            gc.collect()

    def get_model(self, target_class):
        """
        Loads the attack model from disk if memory_efficiency flag is set. Caller is responsible
        for the rest of the life time of the object
        :param target_class:
        :return: attack model for given target class
        """
        if self.memory_efficiency:
            model = self.__model_factory.create()
            model.load(self.path_template.format(target_class))
        else:
            model = self.attack_models[target_class]

        return model


class AttackTrainSetGenerator:
    """
    Class to simplify the generation of training sets to fit MI attack models based on shadow
    training.
    """

    def __init__(self, shadow_group, attack_train_set_size):
        """
        Constructs the generator
        :param shadow_group: a fitted shadow group that should be used for generation
        :param attack_train_set_size: size of the resulting training sets or a dict specifying
        the train size for each attack model separately. As the generator always draws the same
        number of samples from the train and test sets of each shadow model to keep it balanced.
        Hence, if attack_train_set_size is not evenly divisible by 2*shadow_group.size(), the
        returned data set might have less samples.
        """
        self.shadow_group = shadow_group
        self.attack_train_size = attack_train_set_size
        self.X_prior, self.y_prior = shadow_group.get_prior_knowledge()

    def generate_train_set(self, target_class):
        """
        Generates an attack training set w.r.t to a target class
        :param target_class: class to create an attack set for
        :return: tuple of data and labels
        """
        X_shadow, y_shadow = self.shadow_group.get_attack_data_set()

        indices = self.shadow_group.get_train_test_indices()

        posteriors, labels = [], []

        for model_i, (train_indices_i, test_indices_i) in zip(
            self.shadow_group.get_models(), indices
        ):
            X_train, X_test = self._shadow_train_test_split(
                X_shadow, y_shadow, train_indices_i, test_indices_i, target_class
            )

            # in some edge-cases, a shadow model wont receive records for all classes
            if len(X_train) > 0:
                posteriors.append(model_i.predict(X_train))
                labels.append(np.ones(len(X_train)))  # adding ones indicates membership

            if len(X_test) > 0:
                posteriors.append(model_i.predict(X_test))
                labels.append(
                    np.zeros(len(X_test))
                )  # adding zeros indicates no membership

            if self.shadow_group.memory_efficiency:
                model_i.clear_memory()
                del model_i
                gc.collect()

        posteriors = np.concatenate(posteriors)
        labels = np.concatenate(labels)

        return shuffle(posteriors, labels)

    def _shadow_train_test_split(
        self, X_shadow, y_shadow, train_indices, test_indices, target_class
    ):
        """
        Creates train/test for a shadow model containing only samples of one target class.
        The train and test indices are already known, as they were computed by the shadow group
        while training and evaluating the shadow models
        :param X_shadow: the training set used for all shadow models
        :param y_shadow: the labels used for all shadow models
        :param train_indices: the indices of X_shadow used while training a specific shadow model
        :param test_indices: the indices w.r.t. X_shadow for testing a specific shadow model
        :param target_class: class for which samples should be selected
        :return: tuple of shadow train and test set
        """
        mask_train = y_shadow[train_indices] == target_class
        mask_test = y_shadow[test_indices] == target_class

        X_shadow_train = X_shadow[train_indices][mask_train]
        X_shadow_test = X_shadow[test_indices][mask_test]

        if self.X_prior is not None:
            X_shadow_train = np.concatenate(
                [self.X_prior[self.y_prior == target_class], X_shadow_train]
            )

        if type(self.attack_train_size) == dict:
            half = self.attack_train_size[target_class] // (
                2 * self.shadow_group.size()
            )
        else:
            half = self.attack_train_size // (2 * self.shadow_group.size())

        return X_shadow_train[:half], X_shadow_test[:half]

    def _check_train_set_size(self, X_train, X_test):
        raise NotImplementedError()


class AttackTestSetGenerator:
    """
    Class to simplify the generation of test sets to evaluate MI attack models.
    """

    def __init__(self, target_model, test_set_size):
        """
        creates the generator and assigns a target model
        :param target_model:
        :param test_set_size: size of the generated test sets. if none, it will be set during the
        first invocation of generate_test_set
        """
        self.target_model = target_model
        self.test_set_size = test_set_size

    def generate_test_set(self, target_class, X_target, y_target, X_test, y_test):
        """
        Generates a training set and a test set to evaluate attackers regarding a specified class.
        If not set, test_set_size will be inferred form the X_test, X_train parameters.
        :param target_class: attacked class
        :param X_target: training set of the target model
        :param y_target: labels of the trainings set
        :param X_test: test set of the target model
        :param y_test: labels of the test set
        :return: a tuple of instances,labels indicating membership
        """
        mask1 = np.array(y_target) == target_class
        mask2 = np.array(y_test) == target_class
        index_map1 = np.where(mask1)[0]
        index_map2 = np.where(mask2)[0]

        X_target_for_class = np.array(X_target)[mask1]
        X_test_for_class = np.array(X_test)[mask2]
        self._check_test_set_size(X_target, X_test, target_class)

        # sample target train data
        if type(self.test_set_size) is dict:
            test_set_size = self.test_set_size[target_class]
        else:
            test_set_size = self.test_set_size

        half_N = test_set_size // 2
        indices = np.random.choice(len(X_target_for_class), half_N, replace=False)
        X_target_for_class = X_target_for_class[indices]
        in_indices = index_map1[indices]

        # sample test data
        indices = np.random.choice(len(X_test_for_class), half_N, replace=False)
        X_test_for_class = X_test_for_class[indices]
        out_indices = index_map2[indices]

        X_attack_test = np.concatenate(
            [
                self.target_model.predict(X_target_for_class),
                self.target_model.predict(X_test_for_class),
            ]
        )
        y_attack_test = np.concatenate([np.ones(half_N), np.zeros(half_N)])

        return X_attack_test, y_attack_test, (in_indices, out_indices)

    def _check_test_set_size(self, X_train, X_test, target_class):
        """
        Checks if test_set_size is valid. A value error
        is raised if the instances in X_train and X_test are not enough to build a balanced
        test set of the desired size.
        :param X_train:
        :param X_test:
        :param target_class: ignored in base class implementation
        :return:
        """

        if type(self.test_set_size) is dict:
            test_set_size = self.test_set_size[target_class]
        else:
            test_set_size = self.test_set_size

        if test_set_size // 2 > len(X_train) or test_set_size // 2 > len(X_test):
            raise ValueError(
                f"Cannot create a balanced test set of size {self.test_set_size} using"
                f" X_train of length {len(X_train)} and X_test of length {len(X_test)}"
                f" for class {target_class}."
            )

