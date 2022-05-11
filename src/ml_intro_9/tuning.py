import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
import sklearn.metrics
import click
from pathlib import Path
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
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--inner_folds",
    default=3,
    type=int,
    show_default=True,
)
@click.option(
    "--outer_folds",
    default=10,
    type=int,
    show_default=True,
)
@click.option(
    "--model",
    default='KNeighbors',
    type=click.Choice(['KNeighbors','RandomForest'], case_sensitive=False),
    show_default=True,
)
def tune_params(
    dataset_path: Path,
    random_state: int,
    inner_folds: int,
    outer_folds: int,
    model: str
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    x = dataset.drop("Cover_Type", axis=1).to_numpy()
    y = dataset["Cover_Type"].to_numpy()

    space = dict()
    if model == 'KNeighbors':
        space['classifier__n_neighbors'] = list(range(3,20))

    if model == 'RandomForest':
        space['classifier__n_estimators'] = [10, 20, 30, 50, 100, 150, 200, 300]

    metrics = {
        'accuracy':sklearn.metrics.accuracy_score,
        'f1_macro':lambda a, b: sklearn.metrics.f1_score(a, b, average='macro'),
        'f1_micro':lambda a, b: sklearn.metrics.f1_score(a, b, average='micro')}


    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    # enumerate splits
    outer_results = dict()
    for metric in metrics.keys():
        outer_results[metric] = []

    for train_ix, test_ix in cv_outer.split(x, y):
        # split data
        X_train, X_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        # define the model
        pipeline = create_pipeline(
            random_state=random_state,
            use_scaler=True,
            model=model,
            use_PCA=False,
            use_ICA=False,
            model_params={}
        )
        # define search
        search = GridSearchCV(pipeline, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # report progress
        print('>est=%.3f, cfg=%s' % (result.best_score_, result.best_params_))

        # evaluate the model and store the result
        for metric, metric_func in metrics.items():
            value = metric_func(y_test, yhat)
            outer_results[metric].append(value)
            print(f'>  >{metric} = {value}')

    # summarize the estimated performance of the model
    print("--- Summary ---")
    for metric in metrics:
        print('%s: %.3f (%.3f)' % (metric, np.mean(outer_results[metric]), np.std(outer_results[metric])))