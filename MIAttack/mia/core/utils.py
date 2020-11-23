import os
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
from tensorflow.python.keras.backend import get_session
from tensorflow.python.keras.callbacks import Callback
from tensorflow_privacy.privacy.analysis.rdp_accountant import (
    compute_rdp,
    compute_rdp_from_ledger,
    get_privacy_spent,
)


def save_pickle(obj, directory, filename):
    with open(os.path.join(directory, filename), "wb") as f:
        pkl.dump(obj, f)


def load_pickle(directory, filename):
    with open(os.path.join(directory, filename), "rb") as f:
        obj = pkl.load(f)
    return obj


class ResultSaver:
    """
    Convenience class to persist experimental results right away
    """

    def __init__(self, dir):
        """
        Creates a ResultSaver that maintains an internal dict to save results and persist them later
        on. Also, the instance ensures that all model are persisted in a separate subdirectory.
        :param dir: location where results should be persisted
        """

        self.dir = dir
        self.default_file = "results.pkl"
        self.model_dir = os.path.join(dir, "models")
        self.results = {}

        os.makedirs(
            self.model_dir, exist_ok=True
        )  # creates also the root directory in case it doesn't exists

    def save(self, key, new_obj, parents=[], force_persist=True):
        """
        Saves an object in the internal dictionary. The dict hierarchy can be traversed via
        the parents argument.
        :param key: new key
        :param new_obj: new object
        :param parents: parent keys
        :param force_persist: indicates whether the internal dir is persisted immediately
        :return self
        """
        obj = self.results
        for parent in parents:
            obj = obj[parent]
        obj[key] = new_obj

        if force_persist:
            self.persist()

        return self

    def persist(self):
        """
        persists internal results
        :return:
        """
        save_pickle(self.results, self.dir, self.default_file)

    def persist_model(self, model, name):
        """
        Persists the model in ./<root>/models subdirectory
        :param model: model to persist
        :param name: how to name the file
        """
        file = os.path.join(self.model_dir, name) + ".h5"
        model.save(file)


class TerminateOnEpsilon(Callback):
    """Callback that terminates training when the maximum epsilon budget is used."""

    def __init__(
        self, max_epsilon, target_delta, batch_size, noise_multiplier, train_size
    ):
        super(TerminateOnEpsilon, self).__init__()
        self.last_epoch = 0
        self.target_delta = target_delta
        self.max_epsilon = max_epsilon
        self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        self.train_size = train_size

    def on_epoch_end(self, epoch, logs=None):
        try:
            ledger = self.model.mia_model.optimizer._dp_sum_query.ledger
        except AttributeError:
            ledger = None

        epoch += 1
        privacy = PrivacyReport(
            self.noise_multiplier, epoch, self.batch_size, self.train_size, ledger
        )
        epsilon = privacy.get_epsilon_spent(self.target_delta)

        print("Epsilon = {} in epoch {}".format(epsilon, epoch))

        if epsilon >= self.max_epsilon:
            print(
                "Terminated after max epsilon {} was reached or exceeded in epoch {}".format(
                    self.max_epsilon, epoch
                )
            )
            self.model.stop_training = True


class LossAccuracyCallback(Callback):
    """
    Callback to print loss and accuracy at the end of an epoch.
    """

    def __init__(self, validation=False):
        super(LossAccuracyCallback, self).__init__()
        self.validation = validation

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        loss = logs["loss"]
        train_acc = logs["categorical_accuracy"]
        msg = "Epoch {}: Train Loss {:.4f} Train Acc. {:.4f}".format(
            epoch, np.mean([loss]), train_acc
        )
        if self.validation:
            val_loss = logs["val_loss"]
            val_acc = logs["val_categorical_accuracy"]
            msg += " Val. Loss={:.4f}, Val. Accuracy={:.4f}".format(
                np.mean([val_loss]), val_acc
            )

        print(msg)


class PrivacyReport:
    """
    Reports (\eps, \delta)-DP budget spent at a given stage of training using a DP optimizer
    """

    def __init__(self, noise_multiplier, epochs, batch_size, train_size, ledger=None):
        """
        Computes the budget spent by a DP optimizer to train a model in a private manner.
        If no ledger is provided, a deterministic value is computed.
        Otherwise, the actual samples are taken into account.
        :param noise_multiplier: noise multiplier used by the DP optimizer
        :param epochs: training epochs
        :param batch_size: training batch size
        :param train_size: population size of the training data
        :param ledger: ledger object passed to the DP optimizer
        """

        self.noise_multiplier = noise_multiplier
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_size = train_size
        self.ledger = ledger

    def get_epsilon_spent(self, target_delta):
        """
        Computes the epsilon budget spent by a DP optimizer.
        :param target_delta: fixed delta of an (\eps, \delta)-DP guarantee
        :return: epsilon
        """

        rdp, orders = self._get_rdp_and_orders()
        eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=target_delta)
        if opt_order == max(orders) or opt_order == min(orders):
            print(
                "The privacy estimate is likely to be improved by expanding "
                "the set of orders."
            )
        return eps

    def get_delta_spent(self, target_epsilon):
        """
        Computes the epsilon budget spent by a DP optimizer.
        :param target_epsilon: fixed epsilon of an (\eps, \delta)-DP guarantee
        :return: delta
        """

        rdp, orders = self._get_rdp_and_orders()
        _, delta, opt_order = get_privacy_spent(orders, rdp, target_eps=target_epsilon)
        if opt_order == max(orders) or opt_order == min(orders):
            print(
                "The privacy estimate is likely to be improved by expanding "
                "the set of orders."
            )
        return delta

    def _get_rdp_and_orders(self):
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        if self.ledger is None:
            steps = self.epochs * self.train_size // self.batch_size
            sampling_probability = self.batch_size / self.train_size
            rdp = compute_rdp(
                q=sampling_probability,
                noise_multiplier=self.noise_multiplier,
                steps=steps,
                orders=orders,
            )
        else:
            # calculate with ledger
            print("Formatted Ledger")
            formatted_ledger = self.ledger.get_formatted_ledger(get_session())
            rdp = compute_rdp_from_ledger(formatted_ledger, orders)
        return rdp, orders


class CommonArgumentParser(ArgumentParser):
    """
    Parses commonly used arguments for MIApplications
    """

    def __init__(self, prog=None, description=None):
        super().__init__(prog=prog, description=description)

        # basic MI arguments
        self.add_argument("--data_set", type=str)
        self.add_argument("--target_train_size", type=int)
        self.add_argument("--attack_train_size", type=int, default=None)
        self.add_argument("--num_shadows", type=int)
        self.add_argument("--output_dir", type=str)

        # target model arguments
        self.add_argument("--target_epochs", type=int)
        self.add_argument("--target_batch_size", type=int)
        self.add_argument("--target_learning_rate", type=float)
        self.add_argument("--sigma", type=float, default=None)
        self.add_argument("--norm_clip", type=float, default=None)
        self.add_argument("--target_epsilon", type=float, default=None)

        # attack model arguments
        self.add_argument("--attack_epochs", type=int)
        self.add_argument("--attack_batch_size", type=int)
        self.add_argument("--attack_learning_rate", type=float)
