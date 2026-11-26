
from sklearn.preprocessing import StandardScaler

from ..data.schema import FEATURE_COLUMNS
from .encoder import CATEGORICAL_COLUMNS


# Numerical features in NSL-KDD
NUMERICAL_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]


def create_scaler():
    """
    Create a StandardScaler for numerical features.
    """

    scaler = StandardScaler()

    return scaler
