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
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--model",
            'RTYRT',
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output

