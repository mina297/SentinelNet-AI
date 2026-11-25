from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag"
]


def create_encoder():
    """
    Create a One-Hot Encoder for categorical features.
    """

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=int
    )

    return encoder