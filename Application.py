# imported libraries
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st  # For designing application interface
from imblearn.over_sampling import SMOTE  # SMOTE
from sklearn.model_selection import train_test_split  # data split
from sklearn.preprocessing import StandardScaler  # scaling
from sklearn import metrics  # model Evaluation
from sklearn.ensemble import RandomForestClassifier  # model development
from sklearn.metrics import make_scorer, accuracy_score  # for model accuracy

st.write("""
# FRAUD DETECTION SYSTEM
""")
st.write("---")

# import dataset
df = pd.read_csv('Credit card.csv')

# selected variables
new_df = df[['V2', 'V3', 'V4', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'Class']]

st.sidebar.header("Specify User Input Features")


def user_input_features():
    V2 = st.sidebar.number_input("Enter User V2:")
    V3 = st.sidebar.number_input("Enter User V3:")
    V4 = st.sidebar.number_input("Enter User V4:")
    V7 = st.sidebar.number_input("Enter User V7:")
    V9 = st.sidebar.number_input("Enter User V9:")
    V10 = st.sidebar.number_input("Enter User V10:")
    V11 = st.sidebar.number_input("Enter User V11:")
    V12 = st.sidebar.number_input("Enter User V12:")
    V14 = st.sidebar.number_input("Enter User V14:")
    V16 = st.sidebar.number_input("Enter User V16:")
    V17 = st.sidebar.number_input("Enter User V17:")
    V18 = st.sidebar.number_input("Enter User V18:")

    data = {'V2': V2,
            'V3': V3,
            'V4': V4,
            'V7': V7,
            'V9': V9,
            'V10': V10,
            'V11': V11,
            'V12': V12,
            'V14': V14,
            'V16': V16,
            'V17': V17,
            'V18': V18}

    features = pd.DataFrame(data, index=[0])
    return features


data = user_input_features()

st.header("Specified User Input")
st.write(data)
st.write("---")

y = new_df['Class']
# drop the Stroke column
df = new_df.drop(columns=['Class'])
df = pd.concat([data, df], axis=0)

# Scaling the features using Standardization method
Scaler = StandardScaler()
Scaler.fit(df)
Scaled_X = Scaler.transform(df)

# select the first row(User Input)
data = df[:1]
df = df[1:]

# Splitting the dataset into a train set and test set
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=1)

# SMOTE oversampling
SMOTE_oversample = SMOTE(random_state=1)
X_train, y_train = SMOTE_oversample.fit_resample(X_train, y_train.ravel())

# MODELLING
# create object model
RF_model = RandomForestClassifier(n_estimators=400, min_samples_split=100, min_samples_leaf=30, max_features=6, max_depth=11, bootstrap=False)

# fit the model
RF_model.fit(X_train, y_train)

# MODEL APPLICATION
prediction = RF_model.predict(data)
probability = np.round(RF_model.predict_proba(data)*100)

if prediction == 1:
    prediction = "Transaction is fraudulent"
else:
    prediction = "Transaction is not fraudulent"

st.header("Prediction")
st.write(prediction)
st.write("---")

st.header("Prediction Probability")
st.write('0 - Transaction is not fraudulent')
st.write('1 - Transaction is fraudulent')
st.write(probability)
st.write("---")

percent = RF_model.predict(X_test)
accu = np.round(accuracy_score(y_test, percent)*100)

st.write("Prediction Accuracy= ", accu, "%")
st.write("---")
