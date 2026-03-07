import streamlit as st
from streamlit_option_menu import option_menu
from functions import *


with st.sidebar:

    selected = option_menu('Multiple Disease Prediction System',  
                            ['Heart Disease Prediction',
                             'Stroke Risk Prediction',
                             'Diabetes Prediction'],
                            icons= ['heart','activity','person'],
                            default_index=0)
    
if (selected == 'Diabetes Prediction'):
    
    diabetes_pred_func()


if (selected == 'Heart Disease Prediction'):

    heart_pred_func()


if (selected == 'Stroke Risk Prediction'):

    stroke_pred_func()

    
