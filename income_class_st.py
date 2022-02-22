import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
#from secret import access_key, secret_access_key
import joblib
import streamlit as st
import boto3
import tempfile
import json
import requests
from streamlit_lottie import st_lottie_spinner




train_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Income-classification/master/datasets/train.csv')

test_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Income-classification/master/datasets/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()

gdp_data = pd.read_csv('https://raw.githubusercontent.com/semasuka/Income-classification/master/datasets/GDP.csv')

gdp_data.sort_values(by='1990' , inplace=True,ascending=False)

gdp_data.reset_index(inplace=True, drop=True)

gdp_data.rename(columns={'Country Name':'native-country','1990':'GDP_1990'},inplace=True)


def value_cnt_norm_cal(df,feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat

def add_gdp_data(train_copy,test_copy,gdp_data):
        full_data_copy = pd.concat([train_copy,test_copy],ignore_index=True)
        gdp_group = []
        for idx in gdp_data.index:
            if idx <= 65:
                gdp_group.append('High GDP')
            elif idx >= 65 and idx <= 130:
                gdp_group.append('Medium GDP')
            else:
                gdp_group.append('Low GDP')

        # concatenate the gdp_data with the gdp_group list
        gdp_data = pd.concat([gdp_data.rename(columns={'country':'native-country'}), pd.Series(gdp_group, name='GDP Group')], axis=1)
        # we no longer need the GDP column, so let's drop it
        gdp_data.drop(['GDP_1990'],axis=1,inplace=True)

        # we need to merge the gdp_data with X dataframe
        full_data_copy = pd.merge(full_data_copy, gdp_data, on='native-country', how='left')
        # make income_>50K the last column
        new_col_order = [col for col in full_data_copy.columns if col != 'income_>50K'] + ['income_>50K']
        return full_data_copy[new_col_order]


full_data_copy = add_gdp_data(train_copy,test_copy,gdp_data)

train_copy, test_copy = data_split(full_data_copy,0.2)

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self,col_with_outliers = ['age']):
        self.col_with_outliers = col_with_outliers
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if (set(self.col_with_outliers).issubset(X.columns)):
            Q1 = X[self.col_with_outliers].quantile(.25)
            Q3 = X[self.col_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            outlier_condition = (X[self.col_with_outliers] < (Q1 - 1.5 * IQR)) | (X[self.col_with_outliers] > (Q3 + 1.5 * IQR))
            index_to_keep = X[~outlier_condition.any(axis=1)].index
            return X.loc[index_to_keep]
        else:
            print("Columns not found")
            return X

class MissingValHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # drop all the rows with missing values in X
        X.dropna(inplace=True)
        X.reset_index(inplace=True, drop=True)
        return X
# Input the data from streamlit interface and return the GDP group
def get_gdp_group(country_name):
    # To be implemented
    pass

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,col_with_skewness=['age','capital-gain','capital-loss']):
        self.col_with_skewness = col_with_skewness
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if (set(self.col_with_skewness).issubset(X.columns)):
            # Handle skewness with cubic root transformation
            X[self.col_with_skewness] = np.cbrt(X[self.col_with_skewness])
            return X
        else:
            print('One or more skewed columns are not found')
            return X
class OversampleSMOTE(BaseEstimator,TransformerMixin):
    def __init__(self, perform_oversampling = True):
        self.perform_oversampling = perform_oversampling
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # function to oversample the minority class
        if self.perform_oversampling:
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(X.iloc[:,:-1],X.iloc[:,-1])
            X_y_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return X_y_bal
        else:
            print("No oversampling performed")
            return X


def smote_pipeline_fuc(df):
    smote_pipeline = Pipeline([
        ('smote', OversampleSMOTE()) # default: perform_oversampling = True
    ])
    smote_pip_result = smote_pipeline.fit_transform(df.iloc[:-1])
    profile = df.iloc[[-1]]
    smote_pip_result_final = pd.concat([smote_pip_result,profile],ignore_index=True)
    return smote_pip_result_final


def concat_fuc(df_ordinal_minmax, df_onehot, df_target):
    concat_df = pd.concat([df_ordinal_minmax, df_onehot, df_target], axis=1)
    return concat_df


def one_hot_enc_fuc(df):
    columns_to_one_hot_enc = ['race', 'gender', 'workclass', 'occupation','marital-status', 'relationship']
    one_hot_enc = OneHotEncoder()
    one_hot_enc.fit(df[columns_to_one_hot_enc])
    # get the result of the one hot encoding columns names
    cols_names_one_hot_enc = one_hot_enc.get_feature_names_out(columns_to_one_hot_enc)
    # change the array of the one hot encoding to a dataframe with the column names
    one_hot_result_with_names_col = pd.DataFrame(one_hot_enc.transform(df[columns_to_one_hot_enc]).toarray(),columns=cols_names_one_hot_enc)
    return one_hot_result_with_names_col


def ordinal_minmax_scaler_fuc(df):
    columns_to_ordinal_enc = ['education', 'GDP Group']
    columns_to_scale = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    col_transformer = ColumnTransformer([
    ('Ordinal encoder',OrdinalEncoder(),columns_to_ordinal_enc), # ordinal encoding for education and GDP Group because they are ranked
    ('Min max scaler',MinMaxScaler(),columns_to_scale)]) # scaling for age, capital-gain, capital-loss, hours-per-week
    ordinal_minmax_scaler_result = col_transformer.fit_transform(df)
    ordinal_minmax_scaler_result_with_names_col = pd.DataFrame(ordinal_minmax_scaler_result,columns=columns_to_ordinal_enc+columns_to_scale)
    return ordinal_minmax_scaler_result_with_names_col


def extract_target_col(df):
    target = df.iloc[:,-1].to_frame().reset_index(drop=True)
    return target


def initial_pipeline_fuc(df):
    init_pipeline = Pipeline([
        ('Missing values handler', MissingValHandler()), # drop missing values in the whole dataset
        ('Outliers handler', OutlierHandler()),
        ('Skewness handler', SkewnessHandler()), # columns with skewness are 'age','capital-gain','capital-loss'
    ])
    init_pip_result = init_pipeline.fit_transform(df)
    return init_pip_result

def full_pipeline_fuc(df):
    # initial pipeline
    init_pip_result = initial_pipeline_fuc(df)
    #extracting the target variable
    target = extract_target_col(init_pip_result)
    # column transformers to apply ordinal and minmax transformation on specific columns
    ordinal_minmax_result = ordinal_minmax_scaler_fuc(init_pip_result)
    #one hot encoding
    one_hot_enc_result = one_hot_enc_fuc(init_pip_result)
    # concat the result from the ordinal and minmax transformation and one hot encoding with the target variable
    encoded_concat_result = concat_fuc(ordinal_minmax_result,one_hot_enc_result,target)
    # balance the imbalance data with smote function
    smote_pip_result = smote_pipeline_fuc(encoded_concat_result)
    return smote_pip_result




gdp_data = pd.read_csv('datasets/GDP.csv')
gdp_data.sort_values(by='1990' , inplace=True,ascending=False)
gdp_data.reset_index(inplace=True, drop=True)
gdp_data.rename(columns={'Country Name':'native-country','1990':'GDP_1990'},inplace=True)



def gdp_grouping(country_name):
    gdp_group = ''
    for idx,country in enumerate(gdp_data['native-country'],start=0):
        if country == country_name:
            if idx <= 65:
                gdp_group = 'High GDP'
                return gdp_group
            elif idx >= 65 and idx <= 130:
                gdp_group = 'Medium GDP'
                return gdp_group
            else:
                gdp_group = 'Low GDP'
                return gdp_group




def drop_least_useful_ft(prep_data,feat_list):
    X_train_copy_prep_drop_ft = prep_data.drop(feat_list,axis=1)
    return X_train_copy_prep_drop_ft



############################ Streamlit ############################

st.write("""
# Income Classification
This app predicts if your income is high or low than $50000. Just fill in the following information and click on the Predict button.:
""")

# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select your age', value=38, min_value=15, max_value=78, step=1)

#Gender input
st.write("""
## Gender
""")
input_gender = st.radio('Select you gender',['Male','Female'], index=0)


# Workclass input dropdown
st.write("""
## Workclass
""")
work_class_values = list(value_cnt_norm_cal(full_data,'workclass').index)
work_class_key = ['Private sector', 'Self employed (not incorporated)', 'Local government', 'State government', 'Self employed (incorporated)', 'Without work', 'Never worked']
work_class_dict = dict(zip(work_class_key,work_class_values))
input_workclass_key = st.selectbox('Select your workclass', work_class_key)
input_workclass_val = work_class_dict.get(input_workclass_key)


# Education level input dropdown
st.write("""
## Education level
""")
initial_edu_df = full_data[['education','educational-num']].drop_duplicates().sort_values(by='educational-num').reset_index(drop=True)
edu_key = ['Pre-school', '1st to 4th grade', '5th to 6th grade', '7th to 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma', 'High school graduate', 'Some college', 'Associate degree (vocation)','Associate degree (academic)' ,'Bachelor\'s degree', 'Master\'s degree', 'Professional school', 'Doctorate degree']
edu_df = pd.concat([initial_edu_df,pd.DataFrame(edu_key,columns=['education-letter'])],axis=1)
edu_dict = edu_df.set_index('education-letter').to_dict()['educational-num']
input_edu_key = st.selectbox('Select your highest education level', edu_df['education-letter'])
input_edu_val = edu_dict.get(input_edu_key)
input_education = edu_df.iloc[[input_edu_val-1]]['education'].values[0]


# Marital status input dropdown
st.write("""
## Marital status
""")
marital_status_values = list(value_cnt_norm_cal(full_data,'marital-status').index)
marital_status_key = ['Married (civilian spouse)', 'Never married', 'Divorced', 'Separated', 'Widowed', 'Married (abscent spouse)', 'Married (armed forces spouse)']
marital_status_dict = dict(zip(marital_status_key,marital_status_values))
input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
input_marital_status_val = marital_status_dict.get(input_marital_status_key)



#Occupation input dropdown
st.write("""
## Occupation
""")
occupation_values = list(value_cnt_norm_cal(full_data,'occupation').index)
occupation_key = ['Craftman & repair', 'Professional specialty', 'Executive and managerial role', 'Administrative clerk','Sales', 'Other services', 'Machine operator & inspector', 'Transportation & moving', 'Handlers & cleaners', 'Farming & fishing', 'Technical support', 'Protective service', 'Private house service', 'Armed forces']
occupation_dict = dict(zip(occupation_key,occupation_values))
input_occupation_key = st.selectbox('Select your occupation', occupation_dict)
input_occupation_val = occupation_dict.get(input_occupation_key)

# Relationship input dropdown
st.write("""
## Relationship
""")
relationship_values = list(value_cnt_norm_cal(full_data,'relationship').index)
relationship_key = ['Husband', 'Not in a family', 'Own child', 'Not married','Wife', 'Other relative']
relationship_dict = dict(zip(relationship_key,relationship_values))
input_relationship_key = st.selectbox('Select the type of relationship', relationship_dict)
input_relationship_val = relationship_dict.get(input_relationship_key)

# Race input dropdown
st.write("""
## Race
""")
race_values = list(value_cnt_norm_cal(full_data,'race').index)
race_key = ['White', 'Black', 'Asian & pacific islander', 'American first nation','Other']
race_dict = dict(zip(race_key,race_values))
input_race_key = st.selectbox('Select your race', race_dict)
input_race_val = race_dict.get(input_race_key)

# Capital gain input
st.write("""
## Capital gain
""")
input_capital_gain = st.text_input('Enter any capital gain amount',0,help='A capital gain is a profit from the sale of property or an investment.')


# Capital gain input
st.write("""
## Capital loss
""")
input_capital_loss = st.text_input('Enter any capital loss amount',0,help='A capital loss is a loss from the sale of property or an investment when sold for less than the price it was purchased for.')

# Age input slider
st.write("""
## Hours worked per week
""")
input_hours_worked = st.slider('Select the number of hours you work per week', value=40, min_value=0, max_value=110, step=1)

# Country of residence input dropdown
st.write("""
## Country of residence
""")
input_country = st.selectbox('Select your country of residence', gdp_data['native-country'].sort_values())
gdp = gdp_grouping(input_country)

st.markdown('##')
st.markdown('##')
# Button
predict_bt = st.button('Predict')

profile_to_predict = [input_age, input_workclass_val, 0, input_education, input_edu_val, input_marital_status_val, input_occupation_val, input_relationship_val, input_race_val,input_gender, float(input_capital_gain), float(input_capital_loss), input_hours_worked, input_country, gdp,-1.000]

profile_to_predict_df = pd.DataFrame([profile_to_predict],columns=train_copy.columns)

train_copy_with_profile_to_pred = pd.concat([train_copy,profile_to_predict_df],ignore_index=True)







train_copy_prep = full_pipeline_fuc(train_copy)

test_copy_prep = full_pipeline_fuc(test_copy)

X_train_copy_prep = train_copy_prep.iloc[:,:-1]

y_train_copy_prep = train_copy_prep.iloc[:,-1]


X_test_copy_prep = test_copy_prep.iloc[:,:-1]


y_test_copy_prep = test_copy_prep.iloc[:,-1]



train_copy_with_profile_to_pred = full_pipeline_fuc(train_copy_with_profile_to_pred)

profile_to_pred_prep = train_copy_with_profile_to_pred.iloc[-1:,:-1]








rand_forest_least_pred = [
    'occupation_Handlers-cleaners',
    'workclass_Federal-gov',
    'marital-status_Married-AF-spouse',
    'race_Amer-Indian-Eskimo',
    'occupation_Protective-serv',
    'marital-status_Married-spouse-absent',
    'race_Other',
    'workclass_Without-pay',
    'occupation_Armed-Forces',
    'occupation_Priv-house-serv'
]



profile_to_pred_prep_drop_ft = drop_least_useful_ft(profile_to_pred_prep,rand_forest_least_pred)

st.markdown('##')
st.markdown('##')


#Animation function
@st.experimental_memo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_loading_an = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json')


def make_prediction():
    # connect to s3 bucket
    client = boto3.client('s3', aws_access_key_id=st.secrets["access_key"],aws_secret_access_key=st.secrets["secret_access_key"]) # for s3 API keys when deployed on streamlit share
    #client = boto3.client('s3', aws_access_key_id=access_key,aws_secret_access_key=secret_access_key) # for s3 API keys when deployed on locally

    bucket_name = "incomepredbucket"
    key = "rand_forest_clf.sav"

    # load the model from s3 in a temporary file
    with tempfile.TemporaryFile() as fp:
        client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
        fp.seek(0)
        model = joblib.load(fp)

    # prediction from the model on AWS S3
    return model.predict(profile_to_pred_prep_drop_ft)

if predict_bt:

    with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
        final_pred = make_prediction()
    # if final_pred exists, then stop displaying the loading animation
    if final_pred[0] == 1.0:
        st.success('## You most likely make more than 50k')
    else:
        st.error('## You most likely make less than 50k')








