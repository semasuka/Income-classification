
# People with the highest education level, and who are either husbands or wifes makes more money

Predicting if an individual make more or less than 50K using different information about the person.



## Authors

- [@semasuka](https://www.github.com/semasuka)


## Business problem

This an app to predict if someone make more or less than 50k/year using different features. 
This app can be used when that information is not available or is confidential. 
This app can be usefull for a loan application at the bank or car financing application to have a better financial picture of the applicant.
## Data source

- [Kaggle Income classification](https://www.kaggle.com/lodetomasi1995/income-classification)
- [1990 GDP group dataset](https://www.kaggle.com/nitishabharathi/gdp-per-capita-all-countries)
## Methods

- Exploratory data analysis
- Bivariate analysis
- Multivarate correlation
- Feature engineering
- Feature selection
- S3 bucket model hosting
- Model deployment
## Tech Stack

- Python (refer to requirement.txt for the packages used in this projects)
- Streamlit (interace for the model)
- AWS S3 (model storage)


## Quick glance at the results

Most correlated features to the target.

![heatmat](https://i.ibb.co/GtfKkxn/Screen-Shot-2022-01-17-at-3-37-47-PM.png)

Confusion matrix of random forest (Best estimator with the best parameters)

![Confusion matrix](https://i.ibb.co/bHDQPnt/Screen-Shot-2022-01-17-at-3-47-51-PM.png)

ROC curve of random forest (Best estimator with the best parameters)

![ROC curve](https://i.ibb.co/dWc8P7g/Screen-Shot-2022-01-17-at-3-50-54-PM.png)


## Lessons learned & recommendation

Based on the analysis on this project, we found out that education level and type of relationship is the most predictive feature to determine if someone makes more or less than 50K. Order features like Capital gain, hours work and age are also usefull. The least usefull features with the model used to make prediction are: their occuaption and the workclass they belong to.
## Limitation & next step

- Speed: since the model is deployed on S3, it can take some few seconds to load the model. Solution: cache the model with the Streamlit @st.experimental_singleton for faster reload.
- Dataset used: the dataset used is from 1990, inflation has not been taken into consideration and the countries's economies have changed since then. Solution: retrain with a more recent dataset.
- Hyperparameter tuning: used RandomeSearchCV to save time but could improve the accuracy by couple of % with GridSearchCV .
## Run Locally
Initialize git

```bash
git init
```


Clone the project

```bash
git clone https://github.com/semasuka/Income-classification.git
```

Create a conda virtual environment and install all the packages from the requirements.txt

```bash
conda create —name <env_name> —file requirements.txt 
```

Activate the virtual environment

```bash
conda activate <env_name>
```

List all the packages installed

```bash
conda list
```

Start the streamlit server locally

```bash
streamlit run income_class_st.py
```
If you are having issue with streamlit, please follow [this tutorial on how to set up streamlit](https://docs.streamlit.io/library/get-started/installation)

## Deployment on streamlit

To deploy this project on streamlit share, follow these steps:

- first, make sure you upload your files on Github, including a requirements.txt file
- go to [streamlit share](https://share.streamlit.io/)
- login with Github, Google, etc 
- click on new app button
- select the Github repo name, branch, python file with streamlit code
- click advanced settings, select python version 3.9 and add the secret bucket keys if your model is stored on AWS or GCP
- then save and deploy 

## App deployed on Streamlit

![Streamlit GIF](gif_streamlit/gif_streamlit.gif)
## Repository structure


```
├── gif_streamlit                     
│   ├── gif_streamlit.gif           <- gif file used in the README.
│   
│
├── datasets
│   ├── GDP.csv                     <- the data used to feature engineering/enriched the original data.
│   ├── test.csv                    <- the test data.
│   ├── train.csv                   <- the train data.
│   
│
├── .gitignore                      <- used to ignore certain folder and files that won't be commit to git.
│
│ 
├── income_class_profile.html       <- panda profile html file.
│
│ 
├── Income_Classification.ipynb     <- main python notebook where all the analysis and modeling are done.
│ 
│
├── README.md                       <- this readme file.
│ 
│
├── requirements.txt                <- List of all the dependant packages and versions.

```

## License

[MIT](https://choosealicense.com/licenses/mit/)

