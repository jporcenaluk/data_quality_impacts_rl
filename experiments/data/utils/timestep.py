
from dataclasses import dataclass

@dataclass
class Timestep:
    """
    Save important timesteps in a central place (so that they don't get mis-aligned).

    Formula to get end timestep is: (365 * 24) - (24 * (31 + 30 + 31))

    365 days in a year, 24 hours a day - days in Oct, Nov, Dec (31, 30, 31) * 24

    This preserves the last three months as a test / holdout dataset if want.

    :param start_copy: which timestep to start copy from original clean / dirty dataset
    :param end_copy: which timestep to end copy from original clean / dirty dataset
    :param start_train: which timestep to start training from
    :param end_train: which timestep to end training from
    """
    start_copy: int = 0
    end_copy: int = 8760
    start_train: int = 0
    end_train: int = 6552