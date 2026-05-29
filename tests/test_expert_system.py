import os
import pytest
import numpy as np
from degann.expert import ExpertSystemTags, suggest_parameters, execute_pipeline
from degann.expert.tags import (
    EquationType,
    ModelPredictTime,
    DataSize,
    RequiredModelPrecision,
)
from degann.networks import IModel
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace


@pytest.fixture
def train_file_name():
    return "exp_150_train.csv"


@pytest.fixture
def validate_file_name():
    return "exp_150_validate.csv"


@pytest.fixture
def equation_data(train_file_name, validate_file_name):
    test_dir = os.path.dirname(__file__)
    folder_path = os.path.join(test_dir, "data")
    train_data = np.genfromtxt(folder_path + "/" + train_file_name, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((-1, 1))
    train_data_y = train_data_y.reshape((-1, 1))

    validation_data = np.genfromtxt(
        folder_path + "/" + validate_file_name, delimiter=","
    )
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((-1, 1))
    validation_data_y = validation_data_y.reshape((-1, 1))

    return ((train_data_x, train_data_y), (validation_data_x, validation_data_y))


def test_expert_system(equation_data):
    train_data = equation_data[0]
    validation_data = equation_data[1]

    layer_sizes = [10, 15]
    activations = ["sigmoid", "relu"]

    base_params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        losses=["MaxAbsoluteDeviation"],
        layer_sizes=layer_sizes,
        activation_funcs=activations,
        epochs=[5],
        nn_min_depth=1,
        nn_max_depth=1,
    )
    base_config, epoch = next(base_params.iter_configs())
    nn_base = IModel(base_config, net_type="DenseNet")
    nn_base.compile(optimizer=base_config.optimizer, loss_func=base_config.loss_func)
    model_val_loss = nn_base.evaluate(validation_data[0], validation_data[1], verbose=0)

    selector_tags = ExpertSystemTags()
    selector_tags.equation_type = EquationType.EXP
    selector_tags.model_precision = RequiredModelPrecision.MINIMAL
    selector_tags.predict_time = ModelPredictTime.MEDIUM
    selector_tags.data_size = DataSize.MEDIAN

    meta, space = suggest_parameters(train_data, selector_tags)

    meta.threshold = model_val_loss
    meta.iterations = 10

    epochs = list(range(space.min_epoch, space.max_epoch + 1, space.epoch_step))
    expert_params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        losses=["MaxAbsoluteDeviation"],
        layer_sizes=space.layer_sizes,
        activation_funcs=activations,
        epochs=epochs,
        nn_min_depth=space.nn_min_depth,
        nn_max_depth=space.nn_max_depth,
    )

    (
        result_loss,
        result_epoch,
        result_loss_func,
        result_opt,
        result_nn,
    ) = execute_pipeline(
        data=train_data,
        params=expert_params,
        config=meta,
        val_data=validation_data,
        run_grid_search=False,
    )

    model_from_expert_system = IModel.from_dict(result_nn)
    model_from_expert_system.compile(optimizer=result_opt, loss_func=result_loss_func)
    expert_val_loss = model_from_expert_system.evaluate(
        validation_data[0], validation_data[1], verbose=0
    )

    assert expert_val_loss < model_val_loss
