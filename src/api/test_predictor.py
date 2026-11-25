
import pandas as pd
import pytest

from src.api.predictor import (
    prepare_input,
    predict_attack,
    predict_batch,
)

from src.data.loader import load_raw_data
from src.data.schema import FEATURE_COLUMNS
from src.features.scaler import NUMERICAL_COLUMNS


@pytest.fixture(scope="module")
def valid_sample():
    """Load one real sample from the training dataset."""

    train_df, _ = load_raw_data()

    sample = train_df.iloc[0]

    return sample.drop(
        labels=["label", "difficulty"]
    ).to_dict()


# ============================================================
# INPUT VALIDATION TESTS
# ============================================================

def test_prepare_input_accepts_valid_dictionary(valid_sample):

    result = prepare_input(valid_sample)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == FEATURE_COLUMNS
    assert result.shape == (1, 41)


def test_prepare_input_accepts_valid_dataframe(valid_sample):

    result = prepare_input(
        pd.DataFrame([valid_sample])
    )

    assert result.shape == (1, 41)


def test_prepare_input_rejects_missing_features(valid_sample):

    invalid_input = valid_sample.copy()

    invalid_input.pop(FEATURE_COLUMNS[0])

    with pytest.raises(ValueError, match="Missing required features"):

        prepare_input(invalid_input)


def test_prepare_input_rejects_empty_dataframe():

    empty_df = pd.DataFrame(
        columns=FEATURE_COLUMNS
    )

    with pytest.raises(ValueError, match="empty"):

        prepare_input(empty_df)


def test_prepare_input_rejects_missing_values(valid_sample):

    invalid_input = valid_sample.copy()

    invalid_input[NUMERICAL_COLUMNS[0]] = float("nan")

    with pytest.raises(ValueError, match="missing values"):

        prepare_input(invalid_input)


def test_prepare_input_rejects_invalid_numeric_values(valid_sample):

    invalid_input = valid_sample.copy()

    invalid_input[NUMERICAL_COLUMNS[0]] = "not-a-number"

    with pytest.raises(ValueError, match="must contain numeric"):

        prepare_input(invalid_input)


def test_prepare_input_rejects_infinite_values(valid_sample):

    invalid_input = valid_sample.copy()

    invalid_input[NUMERICAL_COLUMNS[0]] = float("inf")

    with pytest.raises(ValueError, match="infinite"):

        prepare_input(invalid_input)


def test_prepare_input_rejects_duplicate_columns(valid_sample):

    df = pd.DataFrame([valid_sample])

    duplicate_df = pd.concat(
        [df, df[[FEATURE_COLUMNS[0]]]],
        axis=1
    )

    with pytest.raises(ValueError, match="duplicate"):

        prepare_input(duplicate_df)


def test_prepare_input_rejects_unsupported_input():

    with pytest.raises(TypeError):

        prepare_input("invalid input")


# ============================================================
# PREDICTION TESTS
# ============================================================

def test_predict_attack_returns_valid_result(valid_sample):

    result = predict_attack(valid_sample)

    assert isinstance(result, dict)

    assert "predicted_attack" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert result["predicted_attack"] in [
        "DoS", "Normal", "Probe", "R2L", "U2R"
    ]

    assert 0 <= result["confidence"] <= 1

    assert set(result["probabilities"].keys()) == {
        "DoS", "Normal", "Probe", "R2L", "U2R"
    }

    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_predict_batch_returns_correct_number_of_results(valid_sample):

    samples = pd.DataFrame(
        [valid_sample, valid_sample, valid_sample]
    )

    results = predict_batch(samples)

    assert len(results) == 3

    for result in results:

        assert "predicted_attack" in result
        assert "confidence" in result
        assert "probabilities" in result


def test_predict_attack_rejects_multiple_samples(valid_sample):

    samples = pd.DataFrame(
        [valid_sample, valid_sample]
    )

    with pytest.raises(ValueError, match="exactly one"):

        predict_attack(samples)
