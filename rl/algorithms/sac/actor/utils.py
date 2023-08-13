from enum import Enum


class TrainMode(Enum):
    STATIC = 0  # no training is permitted
    FINE_TUNING = 1  # load a model and perform fine tuning
    FROM_SCRATCH = 2