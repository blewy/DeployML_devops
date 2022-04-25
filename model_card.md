# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Built as training exercise to develop full CI/CD pipeline with data and model versioning.
The current model was a sklearn Gradient Boosting Classifier with the following hyperparameters:

- n_estimators=300
- learning_rate=0.01,
- max_features=3
- max_depth=3

## Intended Use
This model should be used to predict the expected salary of people based their social demographic information.
The output of the model will classify individual salary band in "<=50K" or ">50k" dollars per year.

## Training Data
The training data was census data provided by UCI Machine Learning repository.
more information following this link: https://archive.ics.uci.edu/ml/datasets/census+income
The data was randomly split in 2 parts (20% validation/80% Training)

## Evaluation Data
The 20% data saved form the random splitting mentioned above

## Metrics
The model was evaluated using multiple metrics, namely:

- Precision: 0.9067796
- Recall: 0.20125
- fbeta: 0.32939 (beta=1)

## Ethical Considerations
According to Aequitas bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model. From Aequitas summary plot we see bias is present in only some of the features and is not consistent across metrics.

## Caveats and Recommendations
This model used default threshold to assign classes of salary, it was not optimized for any kind of metric.
