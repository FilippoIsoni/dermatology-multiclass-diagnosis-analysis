# Dermatology Disease Classification

Comparison of four classification methods — k-NN, decision tree, Naive Bayes, and LDA — for predicting the type of erythemato-squamous skin disease from clinical and histopathological features. Built for the *Machine Learning for Bioengineering* exam (Univ. of Padova, July 2026).

## Dataset

[UCI Dermatology dataset](https://archive.ics.uci.edu/dataset/33/dermatology) (`dermatology.dat`), 358 cases, 20 features:
- 19 clinical/histopathological features on a 0–3 ordinal scale (treated as numerical), including e.g. `clubbing_rete_ridges`, `vacuolisation_damage_basal_layer`, `fibrosis_papillary_dermis`
- 1 age feature
- Target `class`, 6 disease types: psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris

Note: classes are imbalanced (from ~111 cases for class 1 down to ~20 for class 6), which affects per-class sensitivity/specificity for the smaller classes.

## Methods

70/30 train/test split (`caret::createDataPartition`, stratified by class), hyperparameters selected via 10-fold CV on the training set only.

| Model | Test accuracy | Kappa | Notes |
|---|---|---|---|
| Majority classifier (baseline) | 31% | – | Always predicts class 1 |
| k-NN (k=20) | 88% | 0.84 | Low sensitivity for classes 4, 6 |
| Decision tree (cp=0.015, 5 splits) | 89% | 0.86 | Best overall; splits on `clubbing_rete_ridges`, `vacuolisation_damage_basal_layer`, `fibrosis_papillary_dermis` |
| Naive Bayes (kernel density) | 86% | 0.83 | worst on test — struggles with class 4 |
| LDA | 86% | 0.82 | First 3 discriminant directions explain ~91% of between-class variance | plot of the test set observations on the discriminant directions explains the difficulty in the classification of some class on the tst set

## Key findings

- The three features used at the top of the decision tree also separate their respective classes cleanly in density plots and in the LDA discriminant space — the three methods agree on which variables drive the classification.
- Class 4 and class 2 overlap substantially across all models, which explains the recurring drop in class-4 sensitivity.
- LDA's assumption of shared covariance across classes doesn't hold well for class 6 (its cluster is elongated off the discriminant axes), which likely explains why Naive Bayes — the only method not assuming shared covariance — is the only one to reach perfect sensitivity on that class.

## Repository structure

```
├── script.R          # full analysis: data split, kNN, tree, Naive Bayes, LDA, plots
├── report.docx        # written report with commentary and figures
└── dermatology.dat     # dataset (UCI)
```

## Requirements

R with `caret`, `rpart`, `rattle`, `MASS`.

```r
install.packages(c("caret", "rpart", "rattle", "MASS"))
```
