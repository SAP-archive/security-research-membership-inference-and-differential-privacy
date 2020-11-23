import gc
import os

import numpy as np
from mia.core.attack import Attacker, AttackTestSetGenerator, AttackTrainSetGenerator
from mia.core.shadow import ShadowGroup
from mia.core.utils import ResultSaver
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


class MiaApplication:
    """
    Base class for all membership inference applications. Provides base functionality for most
    experiments.
    """

    # default scores
    SCORES = (accuracy_score, precision_score, recall_score)

    def __init__(
        self,
        data_set,
        target_train_size,
        attack_train_size,
        num_shadows,
        target_model_factory,
        attack_model_factory,
        attack_data_set=None,
        prior_knowledge_size=0,
        attack_dataset_sizes="equal",
        memory_efficiency=False,
        shadow_model_factory=None,
        metrics=SCORES,
        target_classes=None,
        output_dir="./",
        verbose=False,
        name="MIAttack",
    ):
        """
        Creates a mia application with several parameters
        :param data_set: data set to train target and shadow models
        :param attack_data_set: data set to draw samples for attacker training from. For ldp target training, this can either be the orig or the ldp domain.
        :param target_train_size: train size for target and shadow models
        :param attack_train_size: size of the attack training set generated for each attack model.
        Recall that an attack model is trained for each target class. Thus, the attack train size
        must be set carefully in order to create no error. If set to None, the maximum attack
        training size will be inferred automatically when calling run().
        :param attack_dataset_sizes: indicates the inference strategy used to infere the number of
        training records and testing records for each attack model. Supported values are
                'equal'     -- to have all attack model trained and tested on the same number of
                               instances, i. e. this restricts all attack models to use only as many
                               instances as provided by the least represented class
                'stratify'  -- attack models for each class receive as many instances as possible,
                               i. e. underrepresented classes might cause some attack models to
                               to be not very stable. If this option is used, the attack_train_size
                               and attack_test_size fields contain a dict with an entry for each
                               class
        :param target_classes: classes or labels to attack. if None, all classes are attacked
        :param num_shadows: number of shadow models
        :param memory_efficiency: makes all shadow models rely on only one physical model to
        save GPU space. Weights are saved in temporary files when switching models
        :param target_model_factory: factory to generate target models that are ready to be trained
        :param shadow_model_factory: factory to generate shadow models that are ready to be trained
        :param attack_model_factory: factory to generate attack models that are ready to be trained
        :param metrics: list of test metrics to evaluate the attack
        :param output_dir: directory where models and results are stored
        :param verbose: False for no outputs, True for some command line information
        :param name: name of the application; default --> MIAttack
        """
        self.data_set = data_set
        self.attack_data_set = attack_data_set
        self.target_train_size = target_train_size
        self.attack_train_size = attack_train_size
        self.prior_knowledge_size = prior_knowledge_size
        self.X_prior = None
        self.y_prior = None
        self.attack_test_size = None
        self.target_classes = target_classes
        self.num_shadows = num_shadows
        self.target_model_factory = target_model_factory
        if shadow_model_factory is not None:
            self.shadow_model_factory = shadow_model_factory
        else:
            self.shadow_model_factory = target_model_factory
        self.attack_model_factory = attack_model_factory
        self.metrics = metrics
        self.result_saver = ResultSaver(output_dir)
        self.output_dir = output_dir
        self.verbose = verbose
        self.name = name
        self._target_indices = None
        self.memory_efficiency = memory_efficiency
        # used to store target temporarily for memory efficiency
        self.target_tmp_path = os.path.join("/", "tmp", "mia_target.h5")

        self.inference_strategy = attack_dataset_sizes
        if self.inference_strategy != "equal" and self.inference_strategy != "stratify":
            raise ValueError(
                f"Attack data size inference strategy {self.inference_strategy} "
                f"not supported..."
            )

    def run(self):
        """
        Runs the membership inference attack. This method is implemented as template method, i. e.
        the steps 'train_target_model', 'train_shadow_group' and others can be re-implemented in
        subclasses if different behavior is desired.
        :return: results stored in ResultSaver object
        """

        self._print("Run '", self.name, "' ...")

        self._print("Load data ...")
        (X_train, y_train), (X_test, y_test) = self.data_set.load_data()

        if self.target_classes is None:
            self.target_classes = np.unique(y_train)

        X_train, y_train = self._extract_prior_knowledge(X_train, y_train)

        (X_target, y_target), (X_shadow, y_shadow) = self._split_target_and_shadow_data(
            X_train, y_train
        )
        self._print("Train target model ...")
        target_model = self._train_target_model(
            X_train=X_target, y_train=y_target, X_test=X_test, y_test=y_test
        )

        self._print("Train shadow models ...")
        shadow_group = self._train_shadow_group(X_train=X_shadow, y_train=y_shadow)

        if self.attack_data_set:
            len_data_set = len(X_train) + len(X_test)
            self._print("Loading different data for attacker training ...")
            (X_train, y_train), (X_test, y_test) = self.attack_data_set.load_data()
            self._validate_attack_data_set_size(
                len(X_train) + len(X_test), len_data_set
            )

            X_train, y_train = self._extract_prior_knowledge(X_train, y_train)
            (
                (X_target, y_target),
                (X_shadow, y_shadow),
            ) = self._resplit_target_and_shadow_data(X_train, y_train)

        shadow_group.set_attack_data_set(X_shadow, y_shadow)

        self._print("Train and evaluate attack model ...")
        self._infer_attack_train_size(shadow_group)

        self._print("Evaluate attack")
        self._infer_attack_test_size(y_target, y_test)

        self._attack(shadow_group, target_model, X_target, y_target, X_test, y_test)

        self.result_saver.persist()

        return self.result_saver.results

    def save_as_result(self, key, new_obj):
        """
        Adds an external python object to the results of the MIAApplication.
        :param key: the key
        :param new_obj: the object associated with the key
        """
        self.result_saver.save(key, new_obj)

    def _extract_prior_knowledge(self, X_train, y_train):

        if self.prior_knowledge_size > self.target_train_size:
            raise ValueError(
                "Expected size of prior knowledge to be smaller than target train size"
            )

        if self.prior_knowledge_size > 0:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=self.prior_knowledge_size, test_size=None
            )

            prior_indices, test_indices = list(sss.split(X_train, y_train))[0]

            self.X_prior = X_train[prior_indices]
            self.y_prior = y_train[prior_indices]

            X_train = X_train[test_indices]
            y_train = y_train[test_indices]

        return X_train, y_train

    def _add_prior_knowledge(self, X, y):
        if self.prior_knowledge_size > 0:
            X = np.concatenate([X, self.X_prior])
            y = np.concatenate([y, self.y_prior])
        return X, y

    def _split_target_and_shadow_data(self, X_train, y_train):
        """
        Splits the original data set into two parts. The first part is used for training the target
        model. The second part is used for training the shadow models. The split guarantees to
        maintain the ratio among labels.
        :param X_train: all samples. Must be a numpy array
        :param y_train: all labels Must be a numpy array
        :return: two tuples with instances and labels in each
        """
        self._validate_train_size(len(X_train))

        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.target_train_size - self.prior_knowledge_size,
            test_size=None,
        )

        target_indices, shadow_indices = list(sss.split(X_train, y_train))[0]

        X_target = X_train[target_indices]
        y_target = y_train[target_indices]

        X_shadow = X_train[shadow_indices]
        y_shadow = y_train[shadow_indices]

        self.result_saver.save("target_train_indices", target_indices)
        self._target_indices = target_indices

        return (X_target, y_target), (X_shadow, y_shadow)

    def _resplit_target_and_shadow_data(self, X_train, y_train):
        """
        Splits a data set into two parts due to the split from _split_target_and_shadow_data().
        :param X_train: all samples. Must be a numpy array
        :param y_train: all labels Must be a numpy array
        :return: two tuples with instances and labels in each
        """
        self._validate_train_size(len(X_train))

        target_indices = self._target_indices
        shadow_indices = [i for i in range(len(X_train)) if i not in target_indices]

        X_target = X_train[target_indices]
        y_target = y_train[target_indices]

        X_shadow = X_train[shadow_indices]
        y_shadow = y_train[shadow_indices]

        return (X_target, y_target), (X_shadow, y_shadow)

    def _train_target_model(self, X_train, y_train, X_test, y_test):
        """
        Trains target model, validates and stores it
        :param X_train: training data set
        :param y_train: training labels
        :param X_test: testing data set
        :param y_test: testing labels
        :return:
        """
        X_train, y_train = self._add_prior_knowledge(X_train, y_train)

        model = self.target_model_factory.create()
        history = model.fit(
            X_train=X_train, y_train=y_train, X_validation=X_test, y_validation=y_test
        )

        self.result_saver.save("target_train_history", history.history)

        train_accuracy = model.evaluate(X_train, y_train)
        test_accuracy = model.evaluate(X_test, y_test)

        self.result_saver.persist_model(model, "target_model")
        self.result_saver.save(
            "target_model_accuracy", {"train": train_accuracy, "test": test_accuracy}
        )

        self._print("train: ", train_accuracy, " test: ", test_accuracy)

        train_confidence = model.predict(X_train)
        test_confidence = model.predict(X_test)

        self.result_saver.save(
            "target_model_confidence",
            {"train": train_confidence, "test": test_confidence},
        )

        if self.memory_efficiency:
            model.save(self.target_tmp_path)
            model.clear_memory()
            del model
            gc.collect()
            return None

        return model

    def _train_shadow_group(self, X_train, y_train):
        """
        Create and fits a shadow group to the given instances and labels
        :param X_train: instances
        :param y_train: labels
        :return:
        """
        shadow_group = ShadowGroup(
            self.num_shadows,
            self.shadow_model_factory,
            self.X_prior,
            self.y_prior,
            self.memory_efficiency,
            os.path.join(self.result_saver.model_dir, "mia_shadow_model_{}.h5"),
        )

        shadow_history = shadow_group.fit(X_train, y_train, self.target_train_size)

        indices = shadow_group.get_train_test_indices()
        self.result_saver.save("shadow_train_indices", indices[:, 0])
        self.result_saver.save("shadow_test_indices", indices[:, 1])
        self.result_saver.save("shadow_train_history", shadow_history)

        return shadow_group

    def _attack(self, shadow_group, target_model, X_target, y_target, X_test, y_test):
        attacker = Attacker(
            self.attack_model_factory, self.target_classes, self.memory_efficiency
        )
        train_generator = AttackTrainSetGenerator(shadow_group, self.attack_train_size)

        X_target, y_target = self._add_prior_knowledge(X_target, y_target)

        scores_summary = {}
        histories = {}
        self.result_saver.save("attack_eval_indices", {})

        for target_class in self.target_classes:
            X_attack_train, y_attack_train = train_generator.generate_train_set(
                target_class
            )

            # the next four lines could be placed outside the loop.
            # However, as with memory_efficiency enabled all models get cleared when model.clear is called during attacker.evaluate.
            # Thus, we have to reinstantiate the target model every iteration
            if self.memory_efficiency:
                target_model = self.target_model_factory.create()
                target_model.load(self.target_tmp_path)

            test_generator = AttackTestSetGenerator(target_model, self.attack_test_size)
            X_attack_test, y_attack_test, indices = test_generator.generate_test_set(
                target_class, X_target, y_target, X_test, y_test
            )
            # save indices
            attack_eval_train_indices = self._target_indices[indices[0]]
            attack_eval_test_indices = indices[1]

            attack_indices_for_target_class = {
                "class": str(target_class),
                "attack_eval_train_indices": attack_eval_train_indices,
                "attack_eval_test_indices": attack_eval_test_indices,
            }

            self.result_saver.save(
                "class_" + str(target_class),
                attack_indices_for_target_class,
                parents=["attack_eval_indices"],
            )

            history = attacker.train_attack_model(
                target_class,
                X_attack_train,
                y_attack_train,
                X_attack_test,
                y_attack_test,
            )

            histories[target_class] = history.history
            model = attacker.get_model(target_class)

            self.result_saver.persist_model(
                model, "attack_model_{}".format(target_class)
            )

            scores, confidence_values = attacker.evaluate_attack(
                X_attack_test, y_attack_test, target_class, self.metrics
            )

            scores_summary[target_class] = self._scores_to_dict(scores)
            self._print("Attack scores for class ", target_class, " ", scores)

            if self.memory_efficiency:
                model.clear_memory()
                del model
                gc.collect()

        self.result_saver.save("attack_scores", scores_summary)
        self.result_saver.save("attack_train_history", histories)

    def _train_attacker(self, shadow_group):
        """
        Creates an attacker and trains it for all target classes using a group of shadow models
        :param shadow_group:
        :return: attacker instance
        """
        attacker = Attacker(
            self.attack_model_factory, self.target_classes, self.memory_efficiency
        )
        generator = AttackTrainSetGenerator(shadow_group, self.attack_train_size)

        attack_history = {}
        for target_class in self.target_classes:
            X_train, y_train = generator.generate_train_set(target_class)
            single_attack_history = attacker.train_attack_model(
                X_train, y_train, target_class
            )
            attack_history[target_class] = single_attack_history.history
            model = attacker.get_model(target_class)
            self.result_saver.persist_model(
                model, "attack_model_{}".format(target_class)
            )

            if self.memory_efficiency:
                model.clear_memory()
                del model
                gc.collect()

        self.result_saver.save("attack_train_history", attack_history)
        return attacker

    def _evaluate_attack(
        self, attacker, target_model, X_target, y_target, X_test, y_test
    ):
        """
        Evaluates the attacker using a target model and the metrics passed to the constructor
        :param attacker: trained attacker
        :param target_model:
        :param X_target: instances used to train the target model
        :param y_target: labels used to train the target model
        :param X_test: instances the target model has not seen so far
        :param y_test: the corresponding labels
        """

        scores_summary = {}
        self.result_saver.save("attack_eval_indices", {})

        for target_class in self.target_classes:

            # the next four lines could be placed outside the loop.
            # However, as with memory_efficiency enabled all models get cleared when model.clear is called during attacker.evaluate.
            # Thus, we have to reinstantiate the target model every iteration
            if self.memory_efficiency:
                target_model = self.target_model_factory.create()
                target_model.load(self.target_tmp_path)

            generator = AttackTestSetGenerator(target_model, self.attack_test_size)

            X_target, y_target = self._add_prior_knowledge(X_target, y_target)
            X_attack, y_attack, indices = generator.generate_test_set(
                target_class, X_target, y_target, X_test, y_test
            )
            # save indices
            attack_eval_train_indices = self._target_indices[indices[0]]
            attack_eval_test_indices = indices[1]

            attack_indices_for_target_class = {
                "class": str(target_class),
                "attack_eval_train_indices": attack_eval_train_indices,
                "attack_eval_test_indices": attack_eval_test_indices,
            }

            self.result_saver.save(
                "class_" + str(target_class),
                attack_indices_for_target_class,
                parents=["attack_eval_indices"],
            )

            scores, confidence_values = attacker.evaluate_attack(
                X_attack, y_attack, target_class, self.metrics
            )

            scores_summary[target_class] = self._scores_to_dict(scores)
            self._print("Attack scores for class ", target_class, " ", scores)

        self.result_saver.save("attack_scores", scores_summary)

        return attacker

    def _scores_to_dict(self, scores):
        """
        Creates a dict from a list of scores with the corresponding metric name as key
        :param scores:
        :return: dict
        """
        obj = {}
        for func, score in zip(self.metrics, scores):
            obj[func.__name__] = score
        return obj

    def _validate_attack_data_set_size(self, attack_data_set_size, data_set_size):
        """
        Check if Attack Data Set and Data Set have the same size
        :param attack_data_set_size: size of attack data set
        :param data_set_size: size of data set
        """
        if not attack_data_set_size == data_set_size:
            raise ValueError(
                f"Size of Data Set and Attack Data Set are {data_set_size} and {attack_data_set_size} but should be equal."
            )

    def _validate_train_size(self, train_data_set_size):
        """
        check if target_train_size is valid. Yet, this is just a first check. Don't rely to much on
        this.
        :param train_data_set_size: size of training set
        """
        if train_data_set_size - self.target_train_size <= 2 * self.target_train_size:
            raise ValueError(
                "The data set for training the shadows (train set minus target train "
                "set) is expected to be greater than twice the train_size. This "
                "guarantees that every shadow model has a disjunct training and test "
                "set of the same size as the target model."
            )

    def _infer_attack_train_size(self, shadow_group):
        """
        Infers a valid training set size for each attack model based on the training data used by
        the shadow group.
        :param shadow_group:
        """
        if self.attack_train_size is None:

            _, y_train = shadow_group.get_group_data_set()
            attack_train_sizes = {}

            for train_indices, test_indices in shadow_group.get_train_test_indices():
                y_shadow_train = y_train[train_indices]  # data used for training
                if self.prior_knowledge_size > 0:
                    y_shadow_train = np.concatenate([y_shadow_train, self.y_prior])
                y_shadow_test = y_train[test_indices]  # data not used for training
                for target_class in self.target_classes:
                    num_train_instances = np.sum(y_shadow_train == target_class)
                    num_test_instances = np.sum(y_shadow_test == target_class)
                    smallest = np.min([num_train_instances, num_test_instances]) * 2
                    if target_class not in attack_train_sizes:
                        attack_train_sizes[target_class] = []
                    attack_train_sizes[target_class].append(smallest)

            for target_class in self.target_classes:
                min_per_shadow = np.min(attack_train_sizes[target_class])
                attack_train_sizes[target_class] = min_per_shadow * self.num_shadows

            if self.inference_strategy == "equal":
                self.attack_train_size = np.min(list(attack_train_sizes.values()))
            else:
                self.attack_train_size = attack_train_sizes

            self.result_saver.save("inferred_attack_train_size", self.attack_train_size)

    def _infer_attack_test_size(self, y_target, y_test):
        """
        Infers the size of the test set for each attack model. Note that the maximum size is chosen.
        :param y_target: the labels used to train the target model
        :param y_test: labels of the test data, i.e. labels of instances the target model did not
        see during training
        """
        attack_test_sizes = {}
        for target_class in self.target_classes:
            # add the numbers of samples for each class
            samples_train = np.sum(y_target == target_class)
            samples_test = np.sum(y_test == target_class)
            attack_test_sizes[target_class] = np.min([samples_train, samples_test]) * 2

        if self.inference_strategy == "equal":
            self.attack_test_size = np.min(list(attack_test_sizes.values()))
        else:
            self.attack_test_size = attack_test_sizes
        self.result_saver.save("inferred_attack_test_size", self.attack_test_size)

    def _print(self, *args, **kwargs):
        """
        Prints only if verbose flag is set
        """
        if self.verbose:
            print(*args, **kwargs)
