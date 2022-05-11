from click.testing import CliRunner
import pytest

from ml_intro_9.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_model(
    runner: CliRunner
) -> None:
    """It fails given invalid model name."""
    result = runner.invoke(
        train,
        [
            "--model",
            'RTYRT',
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output

def test_for_correct_model(
    runner: CliRunner
) -> None:
    """It fails when model not produced or invalid model produced."""
    runner = CliRunner()
    with runner.isolated_filesystem():

        result = runner.invoke(
            train,
            [
                "-d",
                '../tests/sample_train.csv',
                "-s",
                "test_model.joblib"
            ],
        )
    assert result.exit_code == 0
    assert "accuracy 0.99" in result.output
    assert "model saved in " in result.output



