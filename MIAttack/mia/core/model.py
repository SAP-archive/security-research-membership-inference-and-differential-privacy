import copy
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from sklearn.metrics.classification import accuracy_score
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical


class Model(ABC):
    """
    Abstract class to encapsulate a Keras model. Attack and target models must derive from this
    class.
    """

    def __init__(self):
        """
        Creates a model by calling the abstract method _build_model() that should be overridden by
        sub classes in order to use the desired model architecture.
        """

        self.model = self._build_model()

        # this reference is important to access the mia model within keras callbacks
        self.model.mia_model = self

        # default training parameters
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.generators = {"train": None, "validation": None, "predict": None}
        self.callbacks = None
        self.enable_per_example_loss = False

    def set_generators(self, train=None, validation=None, predict=None):
        """

        :param train:
        :param validation:
        :param predict:
        :return:
        """

        if train is not None:
            self.generators["train"] = train

        if validation is not None:
            self.generators["validation"] = validation

        if predict is not None:
            self.generators["predict"] = predict

    def set_training_parameters(
        self,
        epochs=None,
        batch_size=None,
        optimizer=None,
        callbacks=None,
        enable_per_example_loss=None,
    ):
        """
        Sets the respective training parameter. Parameters set to None will not be overridden.
        :param epochs:
        :param batch_size:
        :param optimizer:
        :param callbacks: keras callbacks passed to the fit method
        :param enable_per_example_loss: TODO
        """
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if optimizer is not None:
            self.optimizer = optimizer
        if callbacks is not None:
            self.callbacks = callbacks
        if enable_per_example_loss is not None:
            self.enable_per_example_loss = enable_per_example_loss

    def fit(self, X_train, y_train, X_validation=None, y_validation=None):
        """
        Fits the model to the training data. If a generator was specified via
        set_training_parameters, it will be used. By default, the loss function is the
        cross-entropy.
        :param X_train: training instances
        :param y_train:  training labels
        :param X_validation: validation instances
        :param y_validation: validation labels
        """

        X_train = np.array(X_train)
        y_train = to_categorical(y_train)

        if X_validation is not None and y_validation is not None:

            X_validation, y_validation = self._ensure_validation_data_length(
                X_validation, y_validation
            )

            X_validation = np.array(X_validation)
            y_validation = to_categorical(y_validation)

            if self.generators["validation"] is not None:
                self.generators["validation"].set_data(X_validation, y_validation)
                val_data = self.generators["validation"]
            else:
                val_data = (X_validation, y_validation)

        else:
            val_data = None

        if self.enable_per_example_loss:
            loss = CategoricalCrossentropy(
                from_logits=False, reduction=tf.losses.Reduction.NONE
            )
        else:
            loss = CategoricalCrossentropy(from_logits=False)

        self.model.compile(
            optimizer=self.optimizer, loss=loss, metrics=["categorical_accuracy"]
        )

        if self.generators["train"] is None:
            res = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=val_data,
                shuffle=True,
                verbose=0,
                callbacks=self.callbacks,
            )
        else:
            self.generators["train"].set_data(X_train, y_train)
            res = self.model.fit_generator(
                self.generators["train"],
                epochs=self.epochs,
                validation_data=val_data,
                shuffle=True,
                workers=4,
                verbose=0,
                callbacks=self.callbacks,
            )

        return res

    def _ensure_validation_data_length(self, X_val: list, y_val: list) -> (list, list):
        """
        Ensure that validation data set is divisible by
        :param X_val: validation instances
        :param y_val: validation labels
        :return: adjusted validation data
        """

        validation_length = len(X_val)
        idx = np.arange(validation_length)

        validation_length = validation_length - (validation_length % self.batch_size)

        np.random.shuffle(idx)
        idx = idx[:validation_length]

        return (X_val[idx], y_val[idx])

    def predict(self, X_test):
        """
        Runs the model for every sample in X_test and returns a list of values obtained from
        output neurons.
        :param X_test: a set of test samples
        :return: a list of output values calculated by the model for each input
        """
        if self.generators["predict"] is None:
            if len(X_test) == 0:
                raise ValueError("Empty list passed to predict")
            return self.model.predict(np.array(X_test))

        self.generators["predict"].set_data(X_test, None)
        return self.model.predict_generator(self.generators["predict"])

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on a test set.
        :param X_test: test instances, must be an array of instances, e. g. for one instance
                       [instance]
        :param y_test: test labels
        :return: test accuracy
        """

        prediction = np.argmax(self.predict(X_test), axis=1)
        return accuracy_score(y_test, prediction)

    def save(self, file):
        """
        saves the model at the desired location
        :param file: location where to save the model
        """
        # assumes the graph has been built

        self.model.save_weights(file)

    def load(self, file):
        """
        Loads weights from file into this model
        :param file:
        """
        self.model.load_weights(file)

    @abstractmethod
    def _build_model(self):
        """
        Subclasses are supposed to override this method to define their desired model architecture.
        :return: the architecture
        """
        pass

    def clear_memory(self):
        """
        Clears all the memory in the session this model was created ins
        :return:
        """
        tf.keras.backend.clear_session()


class AttackModel(Model):
    """
    Default attack model
    """

    def __init__(self, input_shape: int):
        """
        Creates a default attack model that should suffice in most cases.
        :param input_shape: number of input neurons == number of outputs of the target model. Type
                            must be int!!!
        """
        self.input_shape = input_shape
        super().__init__()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(64, input_shape=(self.input_shape,)))
        model.add(Activation("relu"))

        num_classes = 2  # an attack model is always a binary classifier
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model


class ModelFactory:
    """
    A generic model factory that always creates the same type of model with the same set of
    parameters.
    """

    def __init__(self, model_class, **kwargs):
        """
        Creates a generic model factory
        :param model_class: the type of models to create
        :param kwargs: constructor arguments for model_class
        """
        self.model_class = model_class
        self.args = kwargs

    def create(self):
        """
        Instantiates a new model instance of type model_class
        :return: model instance
        """
        return self.model_class(**self.args)


class OptimizerFactory:
    """
    Factory to create the same optimizer object over and over again
    """

    def __init__(self, OptimizerClass, **kwargs):
        """
        Sets the optimizer class and the constructor parameters
        :param OptimizerClass:
        :param kwargs:
        """
        self.OptimizerClass = OptimizerClass
        self.args = kwargs

    def create(self):
        """
        Creates an optimizer object with the respective parameters.
        :return:
        """
        return self.OptimizerClass(**self.args)


class SimpleModelFactory(ModelFactory):
    """
    An easy to use model factory that allows to fix and set training parameters via the constructor.
    """

    def __init__(
        self,
        model_class,
        epochs=None,
        batch_size=None,
        optimizer_factory=None,
        generators=None,
        callbacks=None,
        enable_per_example_loss=None,
        **kwargs
    ):
        """
        Takes a list of arguments that are used as trainings parameters for all instances of model
        class.
        :param model_class: Model class
        :param epochs:
        :param batch_size:
        :param optimizer_factory: the optimizer factory to create an optimizer for every model
        :param generators: a dict of generators
        :param callbacks:
        :param kwargs: arguments passed to the constructor of model_class
        """
        super().__init__(model_class=model_class, **kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_factory = optimizer_factory
        self.generators = generators
        self.callbacks = callbacks
        self.enable_per_example_loss = enable_per_example_loss

    def create(self):
        """
        creates an instance of model_class and sets the predefined trainings parameters
        :return: the model instance
        """
        model = super().create()
        optimizer = (
            None if self.optimizer_factory is None else self.optimizer_factory.create()
        )
        if self.generators is not None:
            generators = copy.deepcopy(self.generators)
            model.set_generators(**generators)
        model.set_training_parameters(
            self.epochs,
            self.batch_size,
            optimizer,
            self.callbacks,
            self.enable_per_example_loss,
        )
        return model

    def set_callbacks(self, callbacks):
        """ set callbacks for models created by this factory """
        self.callbacks = callbacks

    def get_callbacks(self):
        """ return callbacks for this model factory """
        return self.callbacks
