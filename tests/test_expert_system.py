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
from degann.networks.topology.configs import DenseNetConfig
from degann.networks.topology.parameter_space import DenseNetParameterSpace
from degann.search_algorithms.nn_code import default_alphabet


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

    base_params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        loss=["MaxAbsoluteDeviation"],
        min_epoch=10,
        max_epoch=10,
        nn_min_length=3,
        nn_max_length=3,
        nn_alphabet=["20", "10", "08"],
    )
    base_config = base_params.create_parameter_space()[0]
    nn_base = IModel(base_config, net_type="DenseNet")
    nn_base.compile(optimizer=base_config.optimizer, loss_func=base_config.loss_func)
    model_val_loss = nn_base.evaluate(validation_data[0], validation_data[1], verbose=0)

    selector_tags = ExpertSystemTags()
    selector_tags.equation_type = EquationType.EXP
    selector_tags.model_precision = RequiredModelPrecision.MINIMAL
    selector_tags.predict_time = ModelPredictTime.MEDIUM
    selector_tags.data_size = DataSize.MEDIAN

    algorithms_parameters = suggest_parameters(tags=selector_tags)
    algorithms_parameters.loss_function = "MaxAbsoluteDeviation"

    nn_min_length = 1
    nn_max_length = 4
    nn_alphabet = default_alphabet

    if selector_tags.equation_type in [
        EquationType.SIN,
        EquationType.MULTIDIM,
        EquationType.UNKNOWN,
    ]:
        nn_max_length += 1

    if selector_tags.predict_time == ModelPredictTime.SHORT:
        nn_max_length = max(nn_min_length, nn_max_length - 1)
        nn_min_length = max(1, nn_min_length - 1)
    elif selector_tags.predict_time == ModelPredictTime.LONG:
        nn_max_length += 1

    expert_params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=[algorithms_parameters.optimizer],
        loss=[algorithms_parameters.loss_function],
        min_epoch=algorithms_parameters.min_train_epoch,
        max_epoch=algorithms_parameters.max_train_epoch,
        nn_min_length=nn_min_length,
        nn_max_length=nn_max_length,
        nn_alphabet=nn_alphabet,
    )

    result_loss, result_nn = execute_pipeline(
        data=train_data,
        params=expert_params,
        parameters={
            "launch_count_random_search": algorithms_parameters.launch_count_random_search,
            "launch_count_simulated_annealing": algorithms_parameters.launch_count_simulated_annealing,
            "iteration_count": algorithms_parameters.iteration_count,
            "loss_threshold": algorithms_parameters.metric_threshold,
        },
        val_data=validation_data,
        run_grid_search=False,
    )

    config = DenseNetConfig(
        block_size=result_nn["block_size"],
        activation_func=["tanh"] * (len(result_nn["block_size"]) + 1),
        optimizer="Adam",
        loss_func="MeanSquaredError",
        num_epoch=100,
        input_size=result_nn["input_size"],
        output_size=result_nn["output_size"],
    )
    model_from_expert_system = IModel(config=config, net_type=result_nn["net_type"])
    model_from_expert_system.from_dict(result_nn)
    model_from_expert_system.compile(optimizer="Adam", loss_func="MaxAbsoluteDeviation")
    expert_val_loss = model_from_expert_system.evaluate(
        validation_data[0], validation_data[1], verbose=0
    )

    assert expert_val_loss < model_val_loss
