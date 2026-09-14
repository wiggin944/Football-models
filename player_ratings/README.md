# GMM Player Rating System

Unsupervised player-rating model designed to let role archetypes emerge from statistical profiles rather than imposing fixed positions.

## Pipeline

1. Convert volume statistics to per-90 rates.
2. Standardise player features.
3. Fit Gaussian Mixture Models across candidate cluster counts.
4. Select the cluster count using silhouette score.
5. Estimate role-specific feature importance from how strongly each cluster differs from the squad average.
6. Combine weighted z-scores with sample-size reliability and diminishing returns to minutes.
7. Produce relative squad ratings on a 0-10 scale.

The model was tested on real squad data, including Mjällby AIF.

## Methods

Python · pandas · NumPy · scikit-learn · Gaussian Mixture Models · clustering · feature weighting
