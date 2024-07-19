import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, confusion_matrix
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

st.title(':red[PCOS Diagnosis Minor Project-II]')

# Functions
def remove_outliers(data, feature):
    Q1 = np.percentile(data[feature], 25, interpolation="midpoint")
    Q3 = np.percentile(data[feature], 75, interpolation="midpoint")
    IQR = Q3 - Q1
    upper = data[feature] >= (Q3 + 1.5 * IQR)
    lower = data[feature] <= (Q1 - 1.5 * IQR)
    return data[~(upper | lower)]

data = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name=1)

# Dropping the unnamed columns
data.drop(data.columns[data.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
data.drop(data.columns[:2], axis=1, inplace=True)

# Convert non-numeric columns to numeric
for feature in data.select_dtypes(include='object').columns:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

data = data.rename(columns=lambda x: x.strip())

# Fill missing values
data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].mean(), inplace=True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0], inplace=True)
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].mean(), inplace=True)
data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].mean(), inplace=True)

# Dropping columns and removing outliers
data.drop(['Weight (Kg)'], axis=1, inplace=True)
data.drop(['Hip(inch)'], axis=1, inplace=True)

features_to_filter = ['BP _Systolic (mmHg)', "Pulse rate(bpm)", "Waist:Hip Ratio", 'BP _Diastolic (mmHg)']
for feature in features_to_filter:
    data = remove_outliers(data, feature=feature)

# Data scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data)
scaled_data.columns = data.columns

# Data balancing
X = scaled_data.drop('PCOS (Y/N)', axis=1)
y = scaled_data['PCOS (Y/N)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Graphs
st.subheader(':green[Age VS Share of Respondents]')

df = pd.DataFrame({
    'Age group': ['<19', '20-29', '30-44', '45-59', '60>'],
    'Percentage': [3.8, 16.81, 11.58, 1.44, 0.55]
})

chart = alt.Chart(df).mark_bar(color='#FFA07A').encode(
    x=alt.X('Age group', title='Age group'),
    y=alt.Y('Percentage', title='Percentage'),
    text=alt.Text('Percentage', format='.1f'),
    color=alt.Color('Age group')
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
)

chart = chart.properties(
    width=alt.Step(40)  # adjust the width of the bars
)

st.altair_chart(chart, use_container_width=True)

with st.expander("Click to show dataset"):
    st.dataframe(data)

st.title(":red[Start your diagnosis]")
col1, col2 = st.columns(2)
left_fields = ['Age(yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)', 'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)', 'Marraige Status (Yrs)', 'Pregnant(Y/N)', 'No. of aborptions', 'I beta-HCG(mIU/mL)', 'II beta-HCG(mIU/mL)', 'FSH(mIU/mL)', 'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)']
right_fields = ['Waist-Hip Ratio', 'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)', 'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)', 'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)']
last_field = ['Endometrium (mm)']

left_inputs = []
right_inputs = []
last_input = []

with st.sidebar:
    model_sel = st.selectbox("ML model: ", options=["Random Forest Classifier", "XGB Classifier", "AdaBoost Classifier", "Logistic Regression"])

with col1:
    st.subheader("Enter Input Data")
    for field in left_fields:
        input_val = st.number_input(label=field)
        left_inputs.append(input_val)

with col2:
    st.subheader("‎")
    for field in right_fields:
        input_val = st.number_input(label=field)
        right_inputs.append(input_val)

last_inp_value = st.number_input(label=last_field[0], key=10)
last_input.append(last_inp_value)

left_inputs.pop(1)  # Remove 'Weight (Kg)' as it was previously dropped
left_inputs.pop(-2)  # Remove 'Hip(inch)' as it was previously dropped

input_vals = np.array([left_inputs + right_inputs + last_input])

# Define a function to train and predict with the selected model
def train_and_predict(model, X_train, y_train, X_test, y_test, input_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    custom_input = model.predict(input_val)
    
    # Evaluation metrics
    confusion = confusion_matrix(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    
    return preds, custom_input, confusion, accuracy, roc_auc, fpr, tpr

predict = st.button("Predict", use_container_width=True)

if predict:
    if model_sel == "Logistic Regression":
        model = LogisticRegression()
    elif model_sel == "Random Forest Classifier":
        model = RandomForestClassifier()
    elif model_sel == "AdaBoost Classifier":
        model = AdaBoostClassifier()
    elif model_sel == "XGB Classifier":
        model = XGBClassifier()

    preds, custom_input, confusion, accuracy, roc_auc, fpr, tpr = train_and_predict(model, X_train, y_train, X_test, y_test, input_vals)
    
    st.subheader("Diagnosis result")
    st.write("The patient is predicted to have PCOS" if custom_input[0] == 1 else "The patient is predicted not to have PCOS")

    # Display Accuracy
    st.write(f"Accuracy: {accuracy:.2f}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Display ROC AUC Curve
    st.subheader("ROC AUC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
