from sklearn.compose import ColumnTransformer

from .encoder import CATEGORICAL_COLUMNS, create_encoder
from .scaler import NUMERICAL_COLUMNS, create_scaler


def create_preprocessor():
    """
    Combine One-Hot Encoding and Standard Scaling.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                create_encoder(),
                CATEGORICAL_COLUMNS
            ),
            (
                "numerical",
                create_scaler(),
                NUMERICAL_COLUMNS
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor