import pickle
import streamlit as st
import pandas as pd




heart_disease_model = pickle.load(open("D:\Multiple Disease Prediction System\heart_trained_model.sav","rb"))
preprocessor = pickle.load(open('D:\Multiple Disease Prediction System\preprocessor.pkl', 'rb'))


diabetes_model = pickle.load(open("D:\Multiple Disease Prediction System\diabetes_model.sav","rb"))
scaler = pickle.load(open('D:\Multiple Disease Prediction System\diabetes_scaler.sav', 'rb'))

stroke_prediction_model = pickle.load(open("D:\Multiple Disease Prediction System\stroke_trained_model.sav","rb"))
preprocessor_stroke = pickle.load(open("D:\Multiple Disease Prediction System\preprocessor_stroke.pkl","rb"))




def heart_pred_func():

    st.title("Heart Disease Prediction using Machine Learning")

    col1, col2 = st.columns(2,gap = "large")

    with col1:

        age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter your age..")

        sex_label = st.radio("Sex", options=["Male", "Female"])

        cp_map = {
            "Typical Angina": 1,
            "Atypical Angina": 2,
            "Non-anginal Pain": 3,
            "Asymptomatic": 4
        }
        cp_label = st.selectbox("Chest Pain Type", options=list(cp_map.keys()))

        bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=None, placeholder= "Enter your Blood Pressure")


        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=None)

        fbs_label = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])



    with col2:

        ekg_map = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Left ventricular hypertrophy": 2
        }
        ekg_label = st.selectbox("Resting EKG Results", options=list(ekg_map.keys()))
        ekg = ekg_map[ekg_label]


        max_hr = st.slider("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)

        exang_label = st.radio("Exercise Induced Angina", options=["Yes", "No"])

        st_dep = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=None, step=0.1)

        slope_map = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        slope_label = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_map.keys()))

        vessels = st.selectbox("Number of Major Vessels (0-3) colored by Flourosopy", options=[0, 1, 2, 3])

        thal_map = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
        thal_label = st.selectbox("Thallium Stress Test Result", options=list(thal_map.keys()))
        

    if st.button("Predict Heart Disease Risk"):

        inputs = [age, sex_label, cp_label, bp, chol, max_hr, thal_label]

        if None in inputs:

            user_inputs = {
                                "Age": age,
                                "Sex": sex_label,
                                "Chest Pain Type": cp_label,
                                "Blood Pressure": bp,
                                "Cholesterol": chol,
                                "Fasting Blood Sugar": fbs_label,
                                "EKG Results": ekg_label,
                                "Max Heart Rate": max_hr,
                                "Exercise Angina": exang_label,
                                "ST Depression": st_dep,
                                "Slope of ST": slope_label
                        }

            missing_fields = [name for name, value in user_inputs.items() if value is None]
    
            if missing_fields:
    
                st.error("⚠️ The following fields are missing:")
                for field in missing_fields:
                    st.write(f"- {field}")




        else:


            input_df = pd.DataFrame([{
            "Age": age,
            "Sex": 1 if sex_label == "Male" else 0,
            "Chest pain type": cp_map[cp_label],
            "BP": bp,
            "Cholesterol": chol,
            "FBS over 120": 1 if fbs_label == "Yes" else 0,
            "EKG results": ekg_map[ekg_label],
            "Max HR": max_hr,
            "Exercise angina": 1 if exang_label == "Yes" else 0,
            "ST depression": st_dep,
            "Slope of ST": slope_map[slope_label],
            "Number of vessels fluro": vessels,
            "Thallium": thal_map[thal_label]
            }])
            
            transformed_input = preprocessor.transform(input_df)

            prediction = heart_disease_model.predict(transformed_input)
        
            if prediction[0] == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")









def diabetes_pred_func():

    st.title("Diabetes Prediction using Machine Learning")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')


    if st.button("Diabetes Test Result"):
    
        input_data = pd.DataFrame([{
            'Pregnancies': float(Pregnancies),
            'Glucose': float(Glucose),
            'BloodPressure': float(BloodPressure),
            'SkinThickness': float(SkinThickness),
            'Insulin': float(Insulin),
            'BMI': float(BMI),
            'DiabetesPedigreeFunction': float(DiabetesPedigreeFunction),
            'Age': float(Age)
        }])

        scaled_features = scaler.transform(input_data)
            
        prediction = diabetes_model.predict(scaled_features)
        
        if prediction[0] == 1:
            st.error("The person is Diabetic")
        else:
            st.success("The person is Not Diabetic")



def stroke_pred_func():

    st.title("Stroke Risk Prediction using Machine Learning")

    col1, col2 = st.columns(2, gap="large")

    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select Gender")
        age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Enter Age")
        hypertension = st.radio("Hypertension (High BP)", ["No", "Yes"], index=None)
        heart_disease = st.radio("Heart Disease History", ["No", "Yes"], index=None)
        ever_married = st.selectbox("Ever Married?", ["Yes", "No"], index=None, placeholder="Select status")

    
    with col2:
        work_type = st.selectbox("Work Type", 
                                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], 
                                index=None, placeholder="Select Work Type")
        residence = st.selectbox("Residence Type", ["Urban", "Rural"], index=None, placeholder="Select Residence")
        avg_glucose = st.number_input("Average Glucose Level", min_value=40.0, max_value=350.0, value=None, placeholder="e.g. 100.5")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=None, placeholder="e.g. 25.4")
        smoking = st.selectbox("Smoking Status", 
                                ["never smoked", "formerly smoked", "smokes", "Unknown"], 
                                index=None, placeholder="Select Smoking Status")
        


    if st.button("Predict Stroke Risk"):

        user_inputs = {
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "smoking_status": smoking
    }

        missing = [key for key, val in user_inputs.items() if val is None]

        if missing:
            st.error(f"⚠️ Missing fields: {', '.join(missing)}")

        else:
            input_df = pd.DataFrame([user_inputs])

            try:

                prediction = stroke_prediction_model.predict(input_df)
                probability = stroke_prediction_model.predict_proba(input_df)[0][1]

                st.divider()
                if prediction[0] == 1:
                    st.error(f"Result: High Risk of Stroke")
                    st.write(f"Probability: {probability:.2%}")
                else:
                    st.success(f"Result: Low Risk of Stroke")
                    st.write(f"Probability: {probability:.2%}")
                
            except Exception as e:
                st.error(f"Error during transformation or prediction: {e}")