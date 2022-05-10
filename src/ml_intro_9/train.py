from pathlib import Path
from joblib import dump

import numpy as np
import pandas as pd
import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--folds",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--model",
    default='KNeighbors',
    type=click.Choice(['KNeighbors','RandomForest'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-PCA",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-ICA",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--n_neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=10,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    folds: int,
    use_scaler: bool,
    use_ica: bool,
    use_pca: bool,
    model: str,
    n_neighbors: int,
    n_estimators: int
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    x = dataset.drop("Cover_Type", axis=1)
    y = dataset["Cover_Type"]

    if model == 'KNeighbors':
        model_params = {
            'n_neighbors': n_neighbors
        }

    if model == 'RandomForest':
        model_params = {
            'n_estimators': n_estimators
        }

    cv = StratifiedKFold(n_splits=folds)

    metrics = ['accuracy', 'f1_macro', 'f1_micro']

    with mlflow.start_run():
        pipeline = create_pipeline(
            random_state=random_state,
            use_scaler=use_scaler,
            model=model,
            use_PCA=use_pca,
            use_ICA=use_ica,
            model_params=model_params
        )
        results = cross_validate(pipeline, x, y, cv=cv, scoring=metrics)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_PCA", use_pca)
        mlflow.log_param("use_ICA", use_ica)
        mlflow.log_param('random state', random_state)
        mlflow.log_param("model", model)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        for key in metrics:
            value = np.mean(results[f'test_{key}'])
            mlflow.log_param(key, value)
            click.echo(f"{key}: {value}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

if __name__ == '__main__':
    train()