import streamlit as st
import income_classification as ic

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
#workclass_array = np.array(ic.X_train_copy_prep)
st.write(ic.X_train_copy_prep)
#st.selectbox('Select your workclass', workclass_array)