import pandas as pd
import click
from pathlib import Path

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-t",
    "--test-path",
    default="tests/sample_train.csv",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--sample_size",
    default=100,
    type=int,
    show_default=True,
)

def make_test_dataset(
    dataset_path: Path,
    test_path: Path,
    random_state: int,
    sample_size: int
) -> None:
    dataset = pd.read_csv(dataset_path)
    t = dataset.sample(sample_size, random_state=random_state)
    t.to_csv(test_path)