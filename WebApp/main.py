import os
import pickle
import streamlit as st
import pandas as pd

MAIN_PATH  = os.path.abspath(os.getcwd()) # Ambil path working directory
PATH_MODEL = os.path.join(MAIN_PATH, "model", "lgbm-1-classification.pkl") # Menggabungkan Path \Path_dir\model\lgbm-1-classification.pkl
print(MAIN_PATH)

# Make a empty data frame
feature = pd.DataFrame({
    'city':[],
    'city_development_index':[],
    'relevent_experience':[],
    'enrolled_university':[],
    'education_level':[],
    'major_discipline':[],
    'experience':[],
    'company_size':[],
    'company_type':[],
    'last_new_job':[],
    'training_hours':[],
})

# Load Model
lgbm = pickle.load(open(PATH_MODEL, 'rb'))

st.title("Loyality Candidate Prediction") # Membuat judul
st.write("""
         The Loyal Candidate Prediction App is a web-based application that helps employers predict the loyalty of their job candidates. 
         The app uses machine learning algorithms to analyze a candidate's professional history and other relevant data points to determine 
         their likelihood of staying with the company for an extended period.""") # Menampilkan text

name = st.text_input("Nama")
img = st.file_uploader("Upload Photo", ['jpg', 'png', 'jpeg'])

personal_information, working_history = st.columns(2)

with personal_information:
    st.subheader('Personal Information')
    col1, col2 = st.columns(2)
    with col1:
        city = st.number_input("City Index",min_value=1, step=1)
        city_fix = f"city_{city}"
    with col2: 
        cdi = st.number_input("City Development Index", min_value=0.0, max_value=1.0, step=0.1)
    
    rel_exp = st.selectbox("Relevan Experience", options=['Has relevent experience', 'No relevent experience'])
    enr_univ = st.selectbox("Enrolment Univercity", options=['no_enrollment', 'Part time course', 'Full time course'])
    edu = st.selectbox("Education", ('Masters', 'Graduate', 'Phd'))
    major_dicipline = st.selectbox("Major Dicipline", ('STEM', 'Business Degree', 'Arts', 'No Major', 'Humanities','Other'))

with working_history:
    st.subheader('Working History')
    exp = st.number_input(label="Experience (year)", min_value=0,step=1)
    exp_fix = '>20' if exp > 20 else '<1' if exp < 1 else str(exp)
    
    cmp_size = st.selectbox("Company Size", ('10/49', '10000+', '100-500', '50-99', '1000-4999', '<10', '500-999', '5000-9999'))
    cmp_type = st.selectbox("Company Type", ('Pvt Ltd', 'Other', 'Early Stage Startup', 'NGO', 'Funded Startup', 'Public Sector'))
    last_job = st.selectbox("Last New Job", ('1', '>4', '2', 'never', '3', '4'))
    training_hour = st.number_input(label="Training Hours", min_value=0, step=1)

pred_process = st.button("Predict",use_container_width=True)

if pred_process:
    feature.loc[0, 'city'] = f"city_{city}"
    feature.loc[0,'city_development_index'] = cdi
    feature.loc[0,'relevent_experience'] = rel_exp
    feature.loc[0,'enrolled_university'] = enr_univ
    feature.loc[0,'education_level'] = edu
    feature.loc[0,'major_discipline'] = major_dicipline
    feature.loc[0,'experience'] = '>20' if exp > 20 else '<1' if exp < 1 else str(exp)
    feature.loc[0,'company_size'] = cmp_size
    feature.loc[0,'company_type'] = cmp_type
    feature.loc[0,'last_new_job'] = last_job
    feature.loc[0,'training_hours'] = training_hour
    
    prob = lgbm.predict_proba(feature)
    
    photo, information = st.columns(2)
    
    with photo:
        st.image(img)
    
    with information:
        st.write("Name: ", name)
        st.write("Relevant Experience: ", rel_exp)
        st.write("Education Level: ", edu)
        st.write("Experience: ", str(exp), " year")
        st.write("Loyality Pred. : ", str(round(prob[:,0][0], 2))) # [[0, 1]] -> [0] -> 0