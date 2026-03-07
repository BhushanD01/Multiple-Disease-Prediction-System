# Multiple-Disease-Prediction-System



### 🚀 Key Features

- Diabetes Prediction: Utilizes a Random Forest Classifier optimized via GridSearchCV to identify diabetic risk.
  
- Stroke Risk Assessment: Implements an automated Scikit-Learn Pipeline that handles complex data preprocessing (Scaling + One-Hot Encoding) for real-time risk calculation.
  
- Heart Disease Diagnostic: Analyzes cardiovascular metrics to predict potential heart issues using a logistic regression model.



### 📁 Project Structure

```text
.
├── models/
│   ├── diabetes_model.sav           # Optimized RF Model for Diabetes Prediction
│   ├── stroke_trained_model.sav     # Optimized RF Model for Stroke Risk Prediction
│   └── heart_trained_model.sav      # LR Model for Heart Disease Prediction
├── multiple_disease_pred.py         # Main Streamlit Application
├── functions.py                     # Web App Functions
└── requirements.txt                 # Dependencies for deployment
```

### ⚙️ Installation & Usage


**1. Clone the repository**
```python
git clone https://github.com/BhushanD01/Multiple-Disease-Prediction-System.git
```
**2. Install dependencies:**
```python
pip install -r requirements.txt
```
**3. Run the app:**
```python
streamlit run multiple_disease_pred.py
```
