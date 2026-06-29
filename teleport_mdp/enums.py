from enum import Enum, EnumMeta


class ExtendedEnum(Enum, metaclass=EnumMeta):
    @classmethod
    def names(cls) -> set[str]:
        """Return set of names."""

        def func(input_enum: ExtendedEnum) -> str:
            return input_enum.name.upper()

        return set(map(func, cls))

    @classmethod
    def values_list(cls) -> list[object]:
        """Returns list of values."""
        return [m.value for m in cls]

    @classmethod
    def values(cls) -> set[object]:
        """Returns set of values."""

        def func(input_enum: ExtendedEnum) -> object:
            return input_enum.value

        return set(map(func, cls))


class ParamType(str, ExtendedEnum):
    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"


class MetricType(str, ExtendedEnum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    LOG_LOSS = "log_loss"
    CONFUSION_MATRIX = "confusion_matrix"


class OptimizationDirection(str, ExtendedEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Algorithm(str, ExtendedEnum):
    """Learning algorithm to run for an experiment."""

    PPO = "ppo"
    Q_LEARNING = "q_learning"
    TMPI = "tmpi"


class Curriculum(str, ExtendedEnum):
    """Teleport-rate curriculum strategy."""

    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class TeleportDistribution(str, ExtendedEnum):
    """How the teleport distribution xi over states is built."""

    UNIFORM = "uniform"
    UNIFORM_NONTERMINAL = "uniform_nonterminal"
    CUSTOM = "custom"
