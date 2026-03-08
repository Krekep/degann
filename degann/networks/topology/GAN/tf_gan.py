from typing import Optional, Dict, Callable, List, Any, Tuple
import tensorflow as tf
from tensorflow import keras

from degann.networks.topology.DenseNet.tf_densenet import TensorflowDenseNet
from degann.networks.topology.GAN.config import GANConfig
from degann.networks import losses, metrics, optimizers


class TensorflowGAN(tf.keras.Model):
    def __init__(
        self,
        config: GANConfig,
        gen_weight=keras.initializers.HeNormal(),
        gen_biases=keras.initializers.Zeros(),
        disc_weight=keras.initializers.HeNormal(),
        disc_biases=keras.initializers.Zeros(),
        is_debug: bool = False,
        **kwargs,
    ):
        super(TensorflowGAN, self).__init__(**kwargs)

        self.config = config
        self.is_debug = is_debug

        self.generator = TensorflowDenseNet(
            config=config.gen_config,
            weight=gen_weight,
            biases=gen_biases,
            is_debug=is_debug,
            name="TFGenerator",
        )

        self.discriminator = TensorflowDenseNet(
            config=config.disc_config,
            weight=disc_weight,
            biases=disc_biases,
            is_debug=is_debug,
            name="TFDiscriminator",
        )

        self.gen_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.disc_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.gen_loss: Optional[Callable] = None
        self.disc_loss: Optional[Callable] = None
        self.metric_real = tf.keras.metrics.Mean(name="metric_disc_real")
        self.metric_fake = tf.keras.metrics.Mean(name="metric_disc_fake")
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

    def custom_compile(
        self,
        gen_rate=1e-2,
        disc_rate=1e-2,
        gen_optimizer="Adam",
        disc_optimizer="Adam",
        gen_loss_func="MeanSquaredError",
        disc_loss_func="MeanSquaredError",
        metric_funcs=None,
        run_eagerly=False,
    ):
        """
        Configures the model for training

        Parameters
        ----------
        gen_rate: float
            learning rate for generator optimizer
        disc_rate: float
            learning rate for discriminator optimizer
        gen_optimizer: str
            name of generator optimizer
        disc_optimizer: str
            name of discriminator optimizer
        gen_loss_func: str
            name of generator loss function
        disc_loss_func: str
            name of discriminator loss function
        metric_funcs: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """

        if metric_funcs is None:
            metric_funcs = []

        self.gen_optimizer = optimizers.get_optimizer(gen_optimizer)(
            learning_rate=gen_rate
        )
        self.disc_optimizer = optimizers.get_optimizer(disc_optimizer)(
            learning_rate=disc_rate
        )
        self.gen_loss = losses.get_loss(gen_loss_func)
        self.disc_loss = losses.get_loss(disc_loss_func)
        m = [metrics.get_metric(metric) for metric in metric_funcs]
        self.compile(
            optimizer="sgd",
            loss="mse",
            metrics=m,
            run_eagerly=run_eagerly,
        )

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Forward pass through generator."""
        return self.generator(inputs, **kwargs)

    def train_step(self, data) -> Dict[str, tf.Tensor]:
        """
        Custom train step for GAN.

        Parameters
        ----------
        data : tuple
            Pair of (x, y_true).

        Returns
        -------
        dict
            Dictionary with loss values and metrics.
        """
        x, y_true = data

        with tf.GradientTape() as disc_tape:
            y_generated = self.generator(x, training=True)

            real_input_disc = tf.concat([x, y_true], axis=1)
            fake_input_disc = tf.concat([x, y_generated], axis=1)

            real_output = self.discriminator(real_input_disc, training=True)
            fake_output = self.discriminator(fake_input_disc, training=True)

            disc_real_loss = self.disc_loss(tf.ones_like(real_output), real_output)
            disc_fake_loss = self.disc_loss(tf.zeros_like(fake_output), fake_output)
            total_disc_loss = (disc_real_loss + disc_fake_loss) / 2.0

        disc_gradients = disc_tape.gradient(
            total_disc_loss, self.discriminator.trainable_variables
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

        with tf.GradientTape() as gen_tape:
            y_generated = self.generator(x, training=True)

            gen_data_loss = self.gen_loss(y_true, y_generated)

            fake_input_disc_gen = tf.concat([x, y_generated], axis=1)
            fake_output = self.discriminator(fake_input_disc_gen, training=True)
            gen_adversarial_loss = self.gen_loss(tf.ones_like(fake_output), fake_output)

            total_gen_loss = gen_data_loss + gen_adversarial_loss

        gen_gradients = gen_tape.gradient(
            total_gen_loss, self.generator.trainable_variables
        )
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        self.metric_real.update_state(real_output)
        self.metric_fake.update_state(fake_output)

        return {
            "gen_loss": total_gen_loss,
            "disc_loss": total_disc_loss,
            "real_score": self.metric_real.result(),
            "fake_score": self.metric_fake.result(),
        }

    def set_name(self, name):
        """Set model name."""
        self._name = name

    def to_dict(self) -> Dict[str, Any]:
        """
        Export neural network to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the model.
        """

        res = {
            "net_type": "TFGAN",
            "name": self.name,
            "config": self.config.to_dict(),
            "generator": self.generator.to_dict(),
            "discriminator": self.discriminator.to_dict(),
        }
        return res

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Restore neural network from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with model configuration.

        Returns
        -------
        TensorflowGAN
            Restored model instance.
        """
        config = GANConfig.from_dict(config_dict["config"])
        model = cls(config=config, **kwargs)
        model.generator.from_dict(config_dict["generator"])
        model.discriminator.from_dict(config_dict["discriminator"])
        return model

    @property
    def get_activations(self) -> Tuple[List[str], List[str]]:
        """
        Get list of activation functions for each layer.

        Returns
        -------
        tuple[list, list]
            Activations for generator and discriminator.
        """
        gen_activations = self.generator.get_activations
        disc_activations = self.discriminator.get_activations
        return gen_activations, disc_activations

    def __str__(self) -> str:
        gen_str = str(self.generator)
        disc_str = str(self.discriminator)
        return f"GAN Model:\nGenerator:\n{gen_str}\nDiscriminator:\n{disc_str}"
