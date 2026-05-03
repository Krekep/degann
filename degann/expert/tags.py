from enum import Enum


class EquationType(Enum):
    SIN = "sin"
    LIN = "lin"
    EXP = "exp"
    LOG = "log"
    GAUSS = "gauss"
    HYPERBOL = "hyperbol"
    CONST = "const"
    SIG = "sig"
    MULTIDIM = "multidim"  # e.g. y(x, t, z)
    UNKNOWN = "unknown"

    @classmethod
    def __iter__(cls):
        values = [
            cls.SIN,
            cls.LIN,
            cls.EXP,
            cls.LOG,
            cls.GAUSS,
            cls.HYPERBOL,
            cls.CONST,
            cls.SIG,
            cls.MULTIDIM,
            cls.UNKNOWN,
        ]
        return values.__iter__()


class RequiredModelPrecision(Enum):
    MAXIMAL = "maximal"
    MEDIAN = "median"
    MINIMAL = "minimal"

    @classmethod
    def __iter__(cls):
        values = [cls.MAXIMAL, cls.MEDIAN, cls.MINIMAL]
        return values.__iter__()


class ModelPredictTime(Enum):
    LONG = "long"  # predict time doesn't matter
    MEDIUM = "medium"  # not too long
    SHORT = "short"  # as fast as possible

    @classmethod
    def __iter__(cls):
        values = [cls.LONG, cls.MEDIUM, cls.SHORT]
        return values.__iter__()


class DataSize(Enum):
    VERY_SMALL = "very small"
    SMALL = "small"
    MEDIAN = "median"
    BIG = "big"
    AUTO = "auto"

    @classmethod
    def __iter__(cls):
        values = [cls.VERY_SMALL, cls.SMALL, cls.MEDIAN, cls.BIG, cls.AUTO]
        return values.__iter__()


class ExpertSystemTags:
    equation_type: EquationType = EquationType.UNKNOWN
    predict_time: ModelPredictTime = ModelPredictTime.LONG
    model_precision: RequiredModelPrecision = RequiredModelPrecision.MAXIMAL
    data_size: DataSize = DataSize.AUTO


def map_value_to_equation_type(value: str) -> EquationType:
    match value:
        case "sin" | "Sin" | "SIN":
            return EquationType.SIN
        case "lin" | "Lin" | "LIN":
            return EquationType.LIN
        case "exp" | "Exp" | "EXP":
            return EquationType.EXP
        case "log" | "Log" | "LOG":
            return EquationType.LOG
        case "gauss" | "Gauss" | "GAUSS":
            return EquationType.GAUSS
        case "hyperbol" | "Hyperbol" | "HYPERBOL":
            return EquationType.HYPERBOL
        case "const" | "Const" | "CONST":
            return EquationType.CONST
        case "sig" | "Sig" | "SIG":
            return EquationType.SIG
        case "multidim" | "Multidim" | "MULTIDIM":
            return EquationType.MULTIDIM
        case _:
            return EquationType.UNKNOWN


def map_value_to_precision(value: str) -> RequiredModelPrecision:
    match value:
        case "maximal" | "Maximal" | "MAXIMAL":
            return RequiredModelPrecision.MAXIMAL
        case "median" | "Median" | "MEDIAN":
            return RequiredModelPrecision.MEDIAN
        case _:
            return RequiredModelPrecision.MINIMAL


def map_value_to_predict(value: str) -> ModelPredictTime:
    match value:
        case "long" | "Long" | "LONG":
            return ModelPredictTime.LONG
        case "medium" | "Medium" | "MEDIUM":
            return ModelPredictTime.MEDIUM
        case _:
            return ModelPredictTime.SHORT


def map_value_to_data_size(value: str) -> DataSize:
    match value:
        case "very small" | "Very small" | "VERY_SMALL":
            return DataSize.VERY_SMALL
        case "small" | "Small" | "SMALL":
            return DataSize.SMALL
        case "medium" | "Median" | "MEDIAN":
            return DataSize.MEDIAN
        case "big" | "Big" | "BIG":
            return DataSize.BIG
        case _:
            return DataSize.AUTO
