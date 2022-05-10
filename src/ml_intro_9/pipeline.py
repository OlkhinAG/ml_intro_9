from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def create_pipeline(
    random_state: int, use_scaler: bool, model: str, use_TSNE: bool, use_PCA: bool, model_params
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_PCA:
        pipeline_steps.append(("PCA", PCA(n_components=0.9, svd_solver='full', random_state=random_state)))

    if use_TSNE:
        pipeline_steps.append(("TSNE", TSNE(n_components=2, random_state=random_state)))

    if model == 'KNeighbors':
        pipeline_steps.append(
            (
                "classifier",
                KNeighborsClassifier(
                   **model_params
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
