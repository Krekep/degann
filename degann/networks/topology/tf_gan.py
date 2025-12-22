from typing import Optional, Dict, Callable
import tensorflow as tf
from tensorflow import keras

from degann.networks.topology.configs import GANConfig
from degann.networks.topology.tf_gan_components import TensorflowGenerator, TensorflowDiscriminator
from degann.networks import losses


class TensorflowGAN(tf.keras.Model):
    def __init__(
        self,
        config: GANConfig,
        gen_weight_init=keras.initializers.HeNormal(),
        gen_bias_init=tf.zeros_initializer(),
        disc_weight_init=keras.initializers.HeNormal(),
        disc_bias_init=tf.zeros_initializer(),
        is_debug: bool = False,
        **kwargs,
    ):
        super(TensorflowGAN, self).__init__(**kwargs)

        self.config = config
        self.is_debug = is_debug

        self.generator = TensorflowGenerator(
            input_size=config.gen_input_size,
            output_size=config.gen_output_size,
            block_sizes=config.gen_block_sizes,
            activation_funcs=config.gen_activation_funcs,
            out_activation=config.gen_out_activation,
            weight_init=gen_weight_init,
            bias_init=gen_bias_init,
            is_debug=is_debug,
            name="TFGenerator",
        )

        self.discriminator = TensorflowDiscriminator(
            input_size=config.disc_input_size,
            output_size=config.disc_output_size,
            block_sizes=config.disc_block_sizes,
            activation_funcs=config.disc_activation_funcs,
            out_activation=config.disc_out_activation,
            weight_init=disc_weight_init,
            bias_init=disc_bias_init,
            is_debug=is_debug,
            name="TFDiscriminator",
        )

        self.gen_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.disc_optimizer: Optional[keras.optimizers.Optimizer] = None
        self.gen_loss: Optional[Callable] = None
        self.disc_loss: Optional[Callable] = None
        self.metric_real = tf.keras.metrics.Mean(name='metric_disc_real')
        self.metric_fake = tf.keras.metrics.Mean(name='metric_disc_fake')

    def custom_compile(
            self,
            gen_optimizer: str = "Adam",
            disc_optimizer: str = "Adam",
            gen_learning_rate: float = 1e-4,
            disc_learning_rate: float = 1e-4,
            gen_loss: str = "MeanSquaredError",
            disc_loss: str = "MeanSquaredError",
    ):
        self.gen_optimizer = keras.optimizers.get(gen_optimizer)
        self.gen_optimizer.learning_rate = gen_learning_rate
        self.disc_optimizer = keras.optimizers.get(disc_optimizer)
        self.disc_optimizer.learning_rate = disc_learning_rate
        self.gen_loss = losses.get_loss(gen_loss)
        self.disc_loss = losses.get_loss(disc_loss)

    def call(self, inputs, **kwargs):
        return self.generator(inputs, **kwargs)

    def train_step(self, data):
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

        disc_gradients = disc_tape.gradient(total_disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            y_generated = self.generator(x, training=True)
            fake_input_disc_gen = tf.concat([x, y_generated], axis=1)
            fake_output = self.discriminator(fake_input_disc_gen, training=True)
            gen_loss = self.gen_loss(tf.ones_like(fake_output), fake_output)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        self.metric_real.update_state(real_output)
        self.metric_fake.update_state(fake_output)

        return {
            "gen_loss": gen_loss,
            "disc_loss": total_disc_loss,
            "real_score": self.metric_real.result(),
            "fake_score": self.metric_fake.result(),
        }

    def to_dict(self, **kwargs):
        res = {
            "net_type": "TFGAN",
            "name": self.name,
            "config": self.config.to_dict(),
            "generator": self.generator.to_dict(),
            "discriminator": self.discriminator.to_dict(),
        }
        return res

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs):
        config = GANConfig.from_dict(config_dict["config"])
        model = cls(config=config, **kwargs)
        model.generator.from_dict(config_dict["generator"])
        model.discriminator.from_dict(config_dict["discriminator"])
        return model

    @property
    def get_activations(self) -> tuple[list, list]:
        gen_activations = self.generator.get_activations
        disc_activations = self.discriminator.get_activations
        return gen_activations, disc_activations

    def __str__(self) -> str:
        gen_str = str(self.generator)
        disc_str = str(self.discriminator)
        return f"GAN Model:\nGenerator:\n{gen_str}\nDiscriminator:\n{disc_str}"
