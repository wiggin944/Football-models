import numpy as np
from sklearn.decomposition import NMF

from .settings import N_NMF, RNG


def fit_nmf(matrix):
    model = NMF(
        N_NMF,
        init="nndsvda",
        max_iter=800,
        random_state=RNG,
    )

    scores = model.fit_transform(matrix)

    component_scale = model.components_.sum(
        axis=1
    )

    component_scale = np.where(
        np.isfinite(component_scale)
        & (component_scale > 1e-12),
        component_scale,
        1.0,
    )

    # Canonicalise components exactly as the
    # reference implementation does.
    model.components_ = (
        model.components_
        / component_scale[:, None]
    )

    scores = (
        scores
        * component_scale[None, :]
    )

    score_total = scores.sum(
        axis=1,
        keepdims=True,
    )

    normalised = np.divide(
        scores,
        score_total,
        out=np.full_like(
            scores,
            1.0 / scores.shape[1],
        ),
        where=score_total > 1e-12,
    )

    return normalised, model, component_scale
