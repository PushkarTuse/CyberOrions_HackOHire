from flask import Flask 
from flask_restful import Api, Resource 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json


def mlnetwork():
    df = pd.read_csv(r"C:\Users\nitya mohta\Downloads\cybsersecDatabase.csv")
    features = ['totalSourceBytes', 'totalDestinationBytes', 'totalDestinationPackets', 'totalSourcePackets', 'sourcePort', 'destinationPort', 'protocolName']
    X = df.loc[:, features]
    y = df['Label']
    le = LabelEncoder()
    X.loc[:, 'protocolName'] = le.fit_transform(X['protocolName'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    predicted_values = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(300)
    data_dict = predicted_values['Predicted'].to_dict()
    json_data = json.dumps(data_dict)
    return(json_data)
