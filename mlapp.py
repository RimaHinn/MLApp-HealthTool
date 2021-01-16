import streamlit as st
import pandas as pd
import pickle

#use the model that was created in DiabetesDetection.ipynb 
model = pickle.load(open("diabetes_random_forrest_model.pkl", 'rb'))

st.set_page_config(page_title="AI HEALTH TOOLS", page_icon=None)
                   
#function for getting user input through streamlit app
def diabetes_user_input():
   
    pregnancies = st.sidebar.slider("Pregancies",0,10) 
    glucose = st.sidebar.slider("Glucose",0,200) 
    blood_pressure = st.sidebar.slider("Blood Pressure",0,122) 
    skin_thickness = st.sidebar.slider("Skin Thickness",0,100)
    insulin= st.sidebar.slider("Insulin", 0,846)
    bmi = st.sidebar.slider("BMI", 0,70)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5)
    age=  st.sidebar.slider("age", 0, 110)
    
    #dictionary for user input data
    user_input_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'blood_p':blood_pressure,
      'skin_thickness':skin_thickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
    }
    input_data = pd.DataFrame(user_input_data, index=[0])
    return input_data

#function that uses the model to predict "1" for Diabetic or "0" fo non Diabetic
def predict_diabetes(user_data):
    
    prediction= model.predict(user_data.to_numpy())
       
    return prediction  

def main():

    #HTML code variables
    html_main_header = """
    <div style="background:#000000;padding:10px">
    <h2 style="color:white;text-align:center;"> AI Health Tools </h2>
    </div>
    """
    # sub header: Diabetes Preciction ML App
    diabetic_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient is Diabetic</h3>
    </div>
    """
    
    not_diabetic_html = """
    <div style="background:#5c92bf ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient is not Diabetic</h3>
    </div>
    """
    
    # Main Header 

    st.markdown(html_main_header, unsafe_allow_html = True)

    # Side Bar Header
    st.sidebar.image('https://venturebeat.com/wp-content/uploads/2020/11/image001.png', use_column_width= True)
        
    #menu item selcction
    
    menu_items= [" ", "Diabetes Predictor", "Pneumonia Predictor",
                 "COVID-19 Predictor", "Heart Disease Predictor", "Eye Disease Predictor"]
    
    choice = st.sidebar.selectbox("MENU", menu_items)   
    
    if choice == "Diabetes Predictor":
        
        # PATIENT DATA
        st.subheader('DIABETES PREDICTOR')
        st.write("User Input:")
        user_data = diabetes_user_input()
        st.write(user_data)
            
       #ML function called and result determined and displayed
        if st.button("Predict"): 
            pred=predict_diabetes(user_data)
        
            if pred == 1:
                st.markdown(diabetic_html, unsafe_allow_html = True)
                st.write(" ")
                #st.write("Prediction Probability:", model.predict_proba(user_data))
        
            else:
                st.markdown(not_diabetic_html, unsafe_allow_html = True)
                st.write(" ")
                #st.write("Prediction Probability:", model.predict_proba(user_data))
    
    elif choice == "Pneumonia Predictor":
        
        # PATIENT DATA
        st.subheader('PNEUMONIA PREDICTOR')
        
    elif choice == "COVID-19 Predictor":
        
        # PATIENT DATA
        st.subheader('COVID-19 PREDICTOR')
        
    elif choice == "Heart Disease Predictor":
        
        # PATIENT DATA
        st.subheader('HEART DISEASE PREDICTOR')
        
    elif choice == "Eye Disease Predictor":
        
        # PATIENT DATA
        st.subheader('EYE DISEASE PREDICTOR')
      
    
            
   
    
if __name__=='__main__':
    main()

