from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import altair as alt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, confusion_matrix

st.title(':red[PCOS Diagnosis Minor Project-II]')   

def remove_outliers(data, feature):
    Q1 = np.percentile(data[feature], 25, interpolation="midpoint")
    Q3 = np.percentile(data[feature], 75, interpolation="midpoint")
    IQR = Q3 - Q1
    upper = data[feature] >= (Q3 + 1.5 * IQR)
    lower = data[feature] <= (Q1 - 1.5 * IQR)
    return data[~(upper | lower)]

data = pd.read_excel("PCOS_data_without_infertility.xlsx", sheet_name=1)

# dropping the unnamed columns
data.drop(data.columns[data.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
data.drop(data.columns[:2], axis=1, inplace=True)

# few columns were non-numeric; we need to convert them to numeric values
for feature in data.select_dtypes(include='object').columns:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

data = data.rename(columns=lambda x: x.strip())

# filling missing values
data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].mean(), inplace=True)
data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].mode()[0], inplace=True)
data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].mean(), inplace=True)
data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].mean(), inplace=True)

# specifying target column and training feature columns
target = 'PCOS (Y/N)'
all_features = list(data.columns[1:])

features_df = data[all_features]
binary_features = features_df.columns[(features_df.max() == 1) & (features_df.min() == 0)]
binary_data = features_df[binary_features]

# dropping highly correlated values
data.drop(['Weight (Kg)'], axis=1, inplace=True)
data.drop(['Hip(inch)'], axis=1, inplace=True)

# removing outliers
features_to_filter = ['BP _Systolic (mmHg)', "Pulse rate(bpm)", "Waist:Hip Ratio", 'BP _Systolic (mmHg)',
                      'BP _Diastolic (mmHg)']
for feature in features_to_filter:
    data = remove_outliers(data, feature=feature)

# data scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data)
scaled_data.columns = data.columns

# data balancing
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

st.header('')

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

def train_and_predict(model, X_train, y_train, X_test, input_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    custom_input = model.predict(input_val)
    return preds, custom_input

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

    preds, custom_input = train_and_predict(model, X_train, y_train, X_test, input_vals)
    st.subheader("Diagnosis result")
    st.write("The patient is predicted to have PCOS" if custom_input[0] == 1 else "The patient is predicted not to have PCOS")
