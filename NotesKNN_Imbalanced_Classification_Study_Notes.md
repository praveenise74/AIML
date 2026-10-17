# KNN and Imbalanced Classification — Beginner Study Notes

> These notes follow the topics in the attached `KNN.ipynb` and add practical clarifications. Learn the intuition first, then use the examples and revision sheet.

## Contents
1. The problem and KNN intuition
2. KNN prediction steps
3. Distance metrics and scaling
4. scikit-learn implementation
5. Weighted KNN
6. Choosing K: bias, variance, outliers
7. Assumptions, strengths, complexity
8. Multiclass classification
9. KNN imputation
10. Imbalanced data and evaluation metrics
11. Class weights, undersampling, oversampling, SMOTE
12. Safe modelling workflow
13. Common mistakes
14. Revision sheet and self-test
15. References

---

## 1. The problem and KNN intuition

The notebook begins with a reseller/customer example. Each reseller has order counts for categories such as Fresh, Milk, Grocery, Frozen, Detergents/Paper, and Delicassen. Historical resellers have known class labels; the goal is to classify a new reseller from their purchasing pattern.

This is **supervised classification**:
- **Features (`X`)** describe each example.
- **Target/label (`y`)** is the class to predict.
- **Training data** contains examples with known labels.
- **Query point** is the new example to classify.

**K-Nearest Neighbors (KNN)** predicts a new point from the labels of the `K` closest training points. Imagine asking, “Which existing customers behave most like this new customer?” If most similar customers are in class 2, class 2 is a sensible prediction.

KNN is called a **lazy** or **instance-based** method because it mostly stores the training examples rather than learning a compact equation during fitting. Most work happens when a prediction is requested.

Applications include similarity search, recommendations, document similarity, outlier detection, and estimating missing values.

**Memory hook:** KNN = **Keep** the examples, find the **Neighbors**, let their labels **vote**.

## 2. KNN prediction steps

1. Choose `K`, the number of neighbors to consult.
2. Calculate the distance from the query point to every training point.
3. Sort points from nearest to farthest.
4. Select the first `K` points.
5. Count their labels (or add weighted votes).
6. Predict the class with the strongest vote.

Example: the three nearest neighbors of a customer are `Churn`, `Churn`, and `Not churn`. With `K=3`, Churn receives two votes and Not churn one, so KNN predicts **Churn**.

### What does K control?
- `K=1`: only the closest point decides; very sensitive to noise.
- Small K: flexible and local, but can overfit.
- Larger K: smoother predictions, but may ignore local structure.
- K too large: the neighborhood can include many unrelated points, causing underfitting.

Odd K is sometimes used in binary classification to reduce tied votes, but it is not a universal rule. Validation performance matters more than whether K is odd.

## 3. Distance metrics: what does “near” mean?

The distance metric defines which examples count as neighbors. For points `x` and `z` with `d` features:

### Euclidean distance
Straight-line distance:

\[
d(x,z)=\sqrt{\sum_{j=1}^{d}(x_j-z_j)^2}
\]

Between `(1, 2)` and `(4, 6)`:

\[
d=\sqrt{(4-1)^2+(6-2)^2}=\sqrt{9+16}=5
\]

### Manhattan distance
Add the absolute coordinate differences:

\[
d(x,z)=\sum_{j=1}^{d}|x_j-z_j|
\]

For the same points, distance is `|4-1| + |6-2| = 3 + 4 = 7`. Think of moving through city blocks, horizontally and vertically.

### Minkowski distance
A general family:

\[
d(x,z)=\left(\sum_{j=1}^{d}|x_j-z_j|^p\right)^{1/p}
\]

- `p=1`: Manhattan distance.
- `p=2`: Euclidean distance.

Other options include Hamming distance for some categorical/binary representations and cosine-based measures for vector direction. Choose a metric that reflects meaningful similarity for the problem.

### Curse of dimensionality
With many features, distances can become less informative: the nearest point may not be meaningfully closer than the others. KNN can struggle in high-dimensional spaces, particularly when features are irrelevant. Feature selection and dimensionality reduction may help, but should be validated.

## 4. Why feature scaling matters

Suppose one feature is Age (18–80) and another is annual spend (1,000–100,000). Without scaling, spend may dominate the distance calculation. KNN may then behave as if spend is the only important feature.

Two common approaches:

**Standardization** (roughly mean 0, standard deviation 1):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

**Min-max scaling** (often maps values to 0–1):

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

**Leakage rule:** fit the scaler on training data only, then transform validation/test data with that fitted scaler. Never fit preprocessing on the full dataset before splitting. Also encode categorical variables thoughtfully: arbitrary integer codes do not automatically represent meaningful distances.

## 5. Implementing KNN with scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
knn.fit(X_train_s, y_train)
val_pred = knn.predict(X_val_s)
print(classification_report(y_val, val_pred, zero_division=0))
print(confusion_matrix(y_val, val_pred))
```

- `n_neighbors=5`: consult five neighbors.
- `metric="euclidean"`: define closeness using Euclidean distance.
- `weights="uniform"`: each neighbor gets an equal vote.
- `fit`: store the training examples and labels.
- `predict`: find neighbors and vote for each new row.
- `stratify=y`: preserve approximate class proportions in the split.

Use validation data to choose K and other settings. Keep the test set untouched until final evaluation.

Example K search:

```python
from sklearn.metrics import f1_score
results = []
for k in range(1, 31):
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train_s, y_train)
    pred = model.predict(X_val_s)
    results.append((k, f1_score(y_val, pred, zero_division=0)))
best_k, best_f1 = max(results, key=lambda item: item[1])
print(best_k, best_f1)
```

For imbalanced data, choose a metric aligned with the goal (such as minority-class recall or F1), not automatically accuracy.

## 6. Weighted KNN

Ordinary KNN gives every one of the K neighbors the same vote. But perhaps two neighbors are extremely close while three others are much farther away.

Example: Class A neighbors are at distances `0.10` and `0.12`; Class B neighbors are at `0.50`, `0.60`, and `0.70`. Uniform voting predicts B (3 votes to 2). A distance-weighted vote may predict A because its examples are much closer.

A common intuition is inverse-distance weighting:

\[
w_i=\frac{1}{d_i}
\]

In scikit-learn:

```python
knn_weighted = KNeighborsClassifier(
    n_neighbors=5, weights="distance", metric="euclidean"
)
knn_weighted.fit(X_train_s, y_train)
```

- `weights="uniform"`: all selected neighbors count equally.
- `weights="distance"`: closer neighbors have more influence.

Distance weighting is a setting to compare during validation, not a guaranteed improvement. Also, this is **not the same** as assigning higher weights to a minority class in a training loss.

## 7. Choosing K: bias, variance, and outliers

**Bias** is error from a model that is too simple to capture the pattern. **Variance** is sensitivity to small changes in training data.

| K choice | Typical behavior | Risk |
|---|---|---|
| Very small K (e.g., 1) | Follows local detail | High variance; overfits noise |
| Moderate K | Balances detail and smoothing | Often a useful starting point |
| Very large K | Strong smoothing | High bias; underfits |

`K=1` can let a single mislabeled or unusual point control a prediction. Increasing K may reduce the influence of one outlier, but can also wash out a genuine small cluster. Weighted voting may help in some cases, but does not automatically solve outliers.

### KNN's practical assumption
**Similar records tend to have similar labels, and the chosen distance metric captures that similarity.** KNN tends to work better when features are useful and scaled, and nearby points genuinely share a pattern. It can struggle with noisy labels, irrelevant features, sparse neighborhoods, high dimensionality, and very large datasets.

### Time and memory
Let `n` be training rows, `d` features, and `q` query rows. With brute-force search, distance calculation costs roughly `O(n × d)` per query, or `O(q × n × d)` for q queries, plus selecting nearest neighbors. Tree-based methods can help in some settings, but often lose advantage in high dimensions. KNN stores training data, so memory grows roughly with `n × d`. Fitting is relatively cheap; prediction may be expensive.

## 8. Multiclass classification

KNN can classify more than two classes. If the five nearest neighbors contain A: 1, B: 3, and C: 1, the prediction is **B**. `KNeighborsClassifier` handles multiclass labels directly. Evaluate performance for each class, not only the overall score; common classes can hide poor results on a rare class.

## 9. KNN imputation: filling missing values

**Imputation** replaces missing values with estimates. If a customer's Milk quantity is missing, find similar customers using available features and estimate the missing value from their known Milk values.

For a numeric feature, a simple explanation is to average neighbors' known values. For categorical data, a mode or categorical-aware approach may be appropriate.

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

preprocess = make_pipeline(MinMaxScaler(), KNNImputer(n_neighbors=5))
X_train_imputed = preprocess.fit_transform(X_train)
```

KNN imputation is distance-based, so scale differences can distort which rows count as neighbors. Fit imputation on training data only to avoid leakage. Categorical variables need suitable handling; arbitrary numeric category codes can create fake distances.

### Checking imputation
- Compare histograms/distributions before and after.
- Compare feature summaries, remembering missingness may be systematic.
- If possible, hide some known values and test reconstruction accuracy.

**Clarification:** the mean before and after imputation does not have to be exactly equal. A changed mean is a signal to investigate, not automatic proof of failure; a similar mean does not prove the imputation is correct.

## 10. Imbalanced data and evaluation metrics

A dataset is **imbalanced** when classes have very different sample counts. Example: 850 customers did not churn (`0`) and 150 churned (`1`). Class 0 is the majority; class 1 is the minority.

Imbalance is common in fraud detection, disease screening, equipment failure, and churn prediction. The rare class may be the one the business most needs to identify.

### Why accuracy can mislead

\[
\text{Accuracy}=\frac{\text{correct predictions}}{\text{all predictions}}
\]

If 950 of 1,000 records are negative, a model that predicts negative for everyone gets **95% accuracy** but finds **zero positives**. Accuracy alone does not tell us whether the rare class was detected.

### Confusion matrix

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actually positive | True Positive (TP) | False Negative (FN) |
| Actually negative | False Positive (FP) | True Negative (TN) |

- **TP:** correctly identified a positive.
- **FN:** missed a positive.
- **FP:** incorrectly flagged a negative as positive.
- **TN:** correctly identified a negative.

### Precision, recall, and F1

\[
\text{Precision}=\frac{TP}{TP+FP}
\]
Precision asks: of the predicted positives, how many were truly positive?

\[
\text{Recall}=\frac{TP}{TP+FN}
\]
Recall asks: of all actual positives, how many did we find?

\[
F1=2\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}=\frac{2TP}{2TP+FP+FN}
\]

- High recall matters when missing a positive is costly.
- High precision matters when false alarms are costly.
- F1 balances precision and recall but does not include true negatives.
- Balanced accuracy, per-class metrics, PR-AUC, and business costs may also be useful.

No single metric is best for every problem. Choose based on the cost of false positives and false negatives.

```python
from sklearn.metrics import confusion_matrix, classification_report
pred = model.predict(X_test_s)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
```

Check the label order when interpreting a confusion matrix; scikit-learn typically uses sorted label order by default.

## 11. Why imbalance affects KNN and logistic regression

### KNN
KNN votes using local neighbors. If the neighborhood is dominated by the majority class, the model tends to predict that class. A large K can make majority dominance more likely because it uses a broader neighborhood. A small K may capture a local minority cluster but can be sensitive to noise. Neither K choice fixes imbalance in every dataset.

### Logistic regression
Logistic regression learns parameters by minimizing a loss over training examples. When the majority class supplies many more examples, its examples can dominate the total loss unless the setup accounts for imbalance. The classifier may then favor the majority class. This is a risk to measure, not a guarantee that every imbalanced model fails.

## 12. Ways to handle imbalanced data

No single technique always wins. Compare approaches using the same validation strategy and the metric that reflects the real objective.

### 12.1 Class weights / weighted loss

Instead of changing the number of rows, give mistakes on the minority class a larger penalty. In a simple example with 850 negatives and 150 positives, one possible relative weighting is `w0=1` and `w1=850/150 ≈ 5.67`. This is a useful intuition, not a universal formula; balanced weighting is also available in scikit-learn.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train_s, y_train)
```

`class_weight="balanced"` assigns weights inversely related to class frequencies in the training data.

**Advantages:** retains original rows; creates no synthetic examples; often a strong baseline.

**Limitations:** no guarantee of improvement; may increase false positives as it finds more positives; evaluate precision, recall, F1, confusion matrix, and business costs.

**Important distinction:** `LogisticRegression(class_weight="balanced")` changes the class contribution to the training loss. `KNeighborsClassifier(weights="distance")` weights neighbors by distance. These are different mechanisms; KNN does not have the same general `class_weight` option.

### 12.2 Random undersampling

Reduce the majority class by selecting a subset of its examples.

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
```

**Advantages:** smaller training set and potentially faster fitting; can work when majority examples are plentiful/redundant.

**Limitations:** removes data and may discard useful information; results can depend on which rows are removed. It is not restricted to datasets with a billion rows—use evidence and validation to decide.

### 12.3 Random oversampling

Increase the minority class by sampling existing minority rows with replacement.

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
```

**Advantages:** preserves the majority examples and is simple to use.

**Limitations:** repeats minority examples rather than adding new information; can increase overfitting to duplicates.

If there are 200 examples in one class and 100 in the other, and you want to retain all original rows, oversample the minority class (the class with 100 examples).

### 12.4 SMOTE: Synthetic Minority Over-sampling Technique

SMOTE uses nearest neighbors among minority examples to create **synthetic** points instead of merely copying rows.

1. Choose a minority point `x`.
2. Find one of its minority neighbors `z`.
3. Choose a random fraction `r` between 0 and 1.
4. Create a point along the line between them:

\[
x_{new}=x+r(z-x)
\]

Example: `x=(2,4)`, `z=(6,8)`, and `r=0.25` gives `x_new=(3,5)`.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
```

**Advantages:** adds varied synthetic examples instead of exact duplicates; may help a model learn the minority region.

**Limitations:** synthetic examples are not guaranteed to be realistic; interpolation can create ambiguous points near class boundaries; standard SMOTE is designed for numeric feature spaces, so categorical data needs a suitable categorical-aware method. SMOTE does not guarantee better results. The attached notebook reports different outcomes for logistic regression and KNN, and notes that some false-positive/false-negative counts increased. Evaluate rather than assuming it will win.

## 13. Safe modelling workflow: prevent leakage

Correct order:
1. Split into train, validation, and test sets.
2. Fit scalers/preprocessors on training data only.
3. Apply over/undersampling or SMOTE **only to training data**.
4. Train the model on the resampled training data.
5. Evaluate on untouched validation/test sets that keep the natural class distribution.

If SMOTE is applied before splitting, synthetic relatives of training examples can appear in validation/test data, making evaluation too optimistic. Resampling the test set also makes the test less representative of real use.

For cross-validation, use an `imblearn` pipeline so each fold resamples only its training fold. Example comparison:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

smote_model = Pipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])
smote_model.fit(X_train, y_train)
pred = smote_model.predict(X_test)
```

For a fair comparison, compare a no-resampling baseline, class weighting, undersampling, oversampling, and SMOTE on the same untouched validation/test data. For KNN, also compare K values and uniform vs distance weighting.

## 14. Common mistakes

1. Skipping scaling when feature ranges differ.
2. Fitting a scaler or imputer on all data before splitting.
3. Tuning K on the test set.
4. Assuming odd K is always best.
5. Using accuracy alone for imbalanced labels.
6. Confusing distance-weighted neighbors with class-weighted loss.
7. Resampling before the train/test split.
8. Resampling the test set.
9. Assuming SMOTE always improves the model.
10. Assuming imputation must preserve the mean exactly.
11. Treating arbitrary category IDs as meaningful numeric distances.
12. Ignoring the business cost of false positives and false negatives.

## 15. Quick revision sheet

| Concept | Remember this |
|---|---|
| KNN | Find K nearest training points, then vote |
| K | Small K = flexible/noisy; large K = smoother/more biased |
| Euclidean | Straight-line distance |
| Manhattan | Sum of absolute coordinate differences |
| Scaling | Prevents one numeric range dominating distance |
| Weighted KNN | Closer neighbors have more voting influence |
| KNN assumption | Similar nearby records tend to share labels |
| Multiclass | Strongest neighbor vote wins |
| KNN imputation | Use similar rows to estimate missing values |
| Imbalanced data | Classes have unequal sample counts |
| Accuracy trap | Majority-only predictions can look good |
| Precision | Of predicted positives, how many were correct? |
| Recall | Of actual positives, how many did we find? |
| F1 | Balances precision and recall |
| Class weights | Increase penalty for minority-class mistakes |
| Undersampling | Remove some majority examples |
| Oversampling | Repeat minority examples |
| SMOTE | Interpolate synthetic minority examples |
| Leakage prevention | Split first; resample training data only |
| Best method | Validate against the actual objective; no method always wins |

### Memory story
Imagine a new customer entering a store:
1. KNN asks the K most similar existing customers.
2. Distance decides who is similar.
3. Scaling stops one measurement from drowning out others.
4. Weighted KNN listens more to the closest customers.
5. K controls how local or broad the opinion is.
6. Imbalance means most examples belong to one group, so a simple vote may ignore a rare group.
7. Class weights, sampling, and SMOTE are different ways to help a model learn the rare class.
8. Precision, recall, F1, and the confusion matrix reveal whether the model is useful.

### Self-test
1. The three nearest labels are A, A, B. What does ordinary KNN predict?
2. Why can one large-scale feature dominate KNN?
3. What may happen when K is too large?
4. With 950 negatives and 50 positives, what accuracy can an always-negative model achieve? How many positives does it find?
5. If missing a positive is costly, should you examine recall?
6. How is random oversampling different from SMOTE?
7. Why should SMOTE be applied only to training data?
8. Does `weights="distance"` mean minority classes get more weight?
9. What is the main risk of undersampling?
10. Why doesn't a similar mean before and after imputation prove that imputation is good?

**Answers:** 1. A. 2. Distance depends on numeric scale. 3. High bias/underfitting. 4. 95% accuracy, zero positives found. 5. Yes. 6. Oversampling repeats examples; SMOTE interpolates synthetic ones. 7. To prevent leakage and overly optimistic evaluation. 8. No, it weights neighbors by distance. 9. Losing useful information. 10. Different distributions can share the same mean; inspect more and validate where possible.

## 16. References

### Primary source
- Attached `KNN.ipynb`: KNN intuition, distance metrics, scikit-learn examples, weighted KNN, bias–variance, outliers, KNN imputation, churn imbalance, class-weighted logistic regression, random over/undersampling, and SMOTE experiments.

### Official documentation
- scikit-learn, `KNeighborsClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- scikit-learn, Nearest Neighbors user guide: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn, `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- scikit-learn, `f1_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- imbalanced-learn, over-sampling and SMOTE: https://imbalanced-learn.org/stable/over_sampling.html
- imbalanced-learn, under-sampling: https://imbalanced-learn.org/stable/under_sampling.html
- imbalanced-learn, sampler API: https://imbalanced-learn.org/stable/introduction.html

> **Study advice:** First explain the neighbor-voting idea in your own words. Then calculate a small distance, explain why scaling matters, interpret a confusion matrix, and compare class weights, over/undersampling, and SMOTE. Understanding those connections makes the details much easier to remember.
