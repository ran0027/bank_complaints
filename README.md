# Bank Complaints Classifier

My README follows the CRISP-DM process model for data science. [Learn more about CRISP-DM.](https://www.datascience-pm.com/crisp-dm-2)

### BUSINESS UNDERSTANDING

Natural language processing is a buzzword in business today. One use case for NLP is to avoid tedious manual labeling of pieces of text or documents by training a machine learning classification algorithm to classify documents for you (after preprocessing the text using NLP techniques to prepare it for modeling.) This is what I have endeavored to do in this repository.

### DATA UNDERSTANDING

The data we are working with was published by the Consumer Financial Protection Bureau on [data.gov](https://catalog.data.gov/dataset/consumer-complaint-database), and downloaded for use in this project on March 22, 2023.

The data is a collection of over 3 million complaints about consumer financial products and services.

A record consists of a date of receipt, a product and sub-product about which the complaint was made, an issue and a sub-issue, a consumer “complaint narrative”, information about the company providing the product or service and any communication with said company, and some other data.

### DATA PREPARATION

Only 1.2 million of the complaints had a “consumer complaint narrative”, which is the text that I used as a feature to train a classification model, so a dataset of about 1.2 million records was used for modeling, including validation and a holdout test set.

To follow my data preparation process, begin in the Jupyter notebook [EDA.ipynb](EDA.ipynb).

1. EDA.ipynb

Upon exploring the data, I found and corrected some discrepancies in the “Product” and “Sub-product” categories for reviews. I saved this corrected data to disc under a new filename ‘categorized_data.csv’.
Example: the product category “Credit Card”, which contained no sub-products, was separate from the product category “Credit card or prepaid card” which included general-purpose credit card as one of its sub-product categories.

Then I worked through the preprocessing techniques that I wanted to apply to the data step-by-step on a small sample of the data.

2. functions.ipynb

Next, I wrote some functions to perform the preprocessing of the data. These functions are stored in the notebook [functions.ipynb](functions.ipynb).

Note: This required the installation of a special package, ipynb, to import these functions from a notebook file into the main modeling notebook rather than a regular python .py file.

### MODELING

The Jupyter notebook [classifier.ipynb](classifier.ipynb) contains the modeling process for this project.

After loading and preprocessing the data, and splitting the data into train-validate-holdout test sets, the training data was vectorized using term frequency - inverse document frequency vectorization with several different settings.

Several types of classifiers were trained on these representations. In addition to tuning standard hyperparameters for each classification algorithm, random undersampling and decomposition techniques (PCA) were also applied to attempt to improve model performance.

### EVALUATION

In this use case, the “best” model was determined by the model’s accuracy score. The best model had a validation accuracy of 79.5% and a holdout test accuracy of 80%.

### DEPLOYMENT

This model has not been deployed.