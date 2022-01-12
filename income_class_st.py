import streamlit as st
import income_classification as ic
import pandas as pd

st.write("""
# Income Classification
This app predicts if your income is high or low than $50000. Just fill in the following information and click on the button Predict.:
""")

# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select your age', value=38, min_value=15, max_value=110, step=1, help='Slide to select your age')
st.write("Age selected",input_age)


# Workclass input dropdown
st.write("""
## Workclass
""")
work_class_values = list(ic.value_cnt_norm_cal(ic.full_data,'workclass').index)
work_class_key = ['Private sector', 'Self employed (not incorporated)', 'Local government', 'State government', 'Self employed (incorporated)', 'Without work', 'Never worked']
work_class_dict = dict(zip(work_class_key,work_class_values))
input_workclass_key = st.selectbox('Select your workclass', work_class_key)


# Education level input dropdown
st.write("""
## Education level
""")
initial_edu_df = ic.full_data[['education','educational-num']].drop_duplicates().sort_values(by='educational-num').reset_index(drop=True)
work_class_key = ['Pre-school', '1st to 4th grade', '5th to 6th grade', '7th to 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma', 'High school graduate', 'Some college', 'Associate degree (vocation)','Associate degree (academic)' ,'Bachelor\'s degree', 'Master\'s degree', 'Professional school', 'Doctorate degree']
edu_df = pd.concat([initial_edu_df,pd.DataFrame(work_class_key,columns=['education-letter'])],axis=1).drop(columns=['education'])
edu_dict = edu_df.set_index('education-letter').to_dict()['educational-num']
input_edu_key = st.selectbox('Select your highest education level', edu_df['education-letter'])



# Marital status input dropdown
st.write("""
## Marital status
""")
marital_status_values = list(ic.value_cnt_norm_cal(ic.full_data,'marital-status').index)
marital_status_key = ['Married (civilian spouse)', 'Never married', 'Divorced', 'Separated', 'Widowed', 'Married (abscent spouse)', 'Married (armed forces spouse)']
marital_status_dict = dict(zip(marital_status_key,marital_status_values))
st.write(marital_status_dict)
input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)


st.write(ic.full_data)