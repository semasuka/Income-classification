import streamlit as st
import income_classification as ic
import pandas as pd

st.write("""
# Income Classification
This app predicts if your income is high or low than $50000. Just fill in the following information and click on the Predict button.:
""")

# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select your age', value=38, min_value=15, max_value=110, step=1)

#Gender input
st.write("""
## Gender
""")
input_gender = st.radio('Select you gender',['Male','Female'], index=0)


# Workclass input dropdown
st.write("""
## Workclass
""")
work_class_values = list(ic.value_cnt_norm_cal(ic.full_data,'workclass').index)
work_class_key = ['Private sector', 'Self employed (not incorporated)', 'Local government', 'State government', 'Self employed (incorporated)', 'Without work', 'Never worked']
work_class_dict = dict(zip(work_class_key,work_class_values))
input_workclass_key = st.selectbox('Select your workclass', work_class_key)
input_workclass_val = work_class_dict.get(input_workclass_key)


# Education level input dropdown
st.write("""
## Education level
""")
initial_edu_df = ic.full_data[['education','educational-num']].drop_duplicates().sort_values(by='educational-num').reset_index(drop=True)
edu_key = ['Pre-school', '1st to 4th grade', '5th to 6th grade', '7th to 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma', 'High school graduate', 'Some college', 'Associate degree (vocation)','Associate degree (academic)' ,'Bachelor\'s degree', 'Master\'s degree', 'Professional school', 'Doctorate degree']
edu_df = pd.concat([initial_edu_df,pd.DataFrame(edu_key,columns=['education-letter'])],axis=1).drop(columns=['education'])
edu_dict = edu_df.set_index('education-letter').to_dict()['educational-num']
input_edu_key = st.selectbox('Select your highest education level', edu_df['education-letter'])
input_edu_val = edu_dict.get(input_edu_key)


# Marital status input dropdown
st.write("""
## Marital status
""")
marital_status_values = list(ic.value_cnt_norm_cal(ic.full_data,'marital-status').index)
marital_status_key = ['Married (civilian spouse)', 'Never married', 'Divorced', 'Separated', 'Widowed', 'Married (abscent spouse)', 'Married (armed forces spouse)']
marital_status_dict = dict(zip(marital_status_key,marital_status_values))
input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
input_marital_status_val = marital_status_dict.get(input_marital_status_key)



# Occupation input dropdown
st.write("""
## Occupation
""")
occupation_values = list(ic.value_cnt_norm_cal(ic.full_data,'occupation').index)
occupation_key = ['Craftman & repair', 'Professional specialty', 'Executive and managerial role', 'Administrative clerk','Sales', 'Other services', 'Machine operator & inspector', 'Transportation & moving', 'Handlers & cleaners', 'Farming & fishing', 'Technical support', 'Protective service', 'Private house service', 'Armed forces']
occupation_dict = dict(zip(occupation_key,occupation_values))
input_occupation_key = st.selectbox('Select your occupation', occupation_dict)
input_occupation_val = occupation_dict.get(input_occupation_key)

# Relationship input dropdown
st.write("""
## Relationship
""")
relationship_values = list(ic.value_cnt_norm_cal(ic.full_data,'relationship').index)
relationship_key = ['Husband', 'Not in a family', 'Own child', 'Not married','Wife', 'Other relative']
relationship_dict = dict(zip(relationship_key,relationship_values))
input_relationship_key = st.selectbox('Select the type of relationship', relationship_dict)
input_relationship_val = relationship_dict.get(input_relationship_key)

# Race input dropdown
st.write("""
## Race
""")
race_values = list(ic.value_cnt_norm_cal(ic.full_data,'race').index)
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


# Button
predict = st.button('Predict')

profile_to_predict = [input_age,input_gender, input_workclass_val, input_edu_val, input_marital_status_val, input_occupation_val, input_relationship_val, input_race_val, input_capital_gain, input_capital_loss, input_hours_worked]

st.write(profile_to_predict)

if predict:
    # run the prediction
    pass