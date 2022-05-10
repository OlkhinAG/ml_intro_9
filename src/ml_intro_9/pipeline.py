from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    use_scaler: bool, model: str, random_state: int, model_params
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if model == 'KNeighbors':
        pipeline_steps.append(
            (
                "classifier",
                KNeighborsClassifier(
                    random_state=random_state, **model_params
                ),
            )
        )

    if model == 'RandomForest':
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state, **model_params
                ),
            )
        )

    return Pipeline(steps=pipeline_steps)
