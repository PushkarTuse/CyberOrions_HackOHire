from flask import Flask 
from flask_restful import Api, Resource 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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


def attacktype():
    data_url = 'https://archive.ics.uci.edu/static/public/942/data.csv'
    data = pd.read_csv(data_url)
    data.dropna(inplace=True)
    label_encoder = LabelEncoder()
    categorical_columns = ['proto', 'service']
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    X = data.drop('Attack_type', axis=1)
    y = data['Attack_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    data1 = pd.read_csv(r"C:\Users\nitya mohta\Downloads\cybersecurity_attacks.csv")
    imputer = KNNImputer()
    data1[data1.select_dtypes(include=['int', 'float']).columns] = imputer.fit_transform(data1.select_dtypes(include=['int', 'float']))
    data1[data1.select_dtypes(include=['object']).columns] = data1[data1.select_dtypes(include=['object']).columns].fillna(data1[data1.select_dtypes(include=['object']).columns].mode().iloc[0])
    label_encoders = {}
    for column in data1.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data1[column] = label_encoders[column].fit_transform(data1[column])
    A = data1.drop(columns=['Attack Type'])
    b = data1['Attack Type']
    chi2_scores, _ = chi2(A, b)
    feature_chi2 = pd.DataFrame({'Feature': A.columns, 'Chi2 Score': chi2_scores})
    feature_chi2 = feature_chi2.sort_values(by='Chi2 Score', ascending=False)
    chi2_threshold = 1000 
    significant_features = feature_chi2[feature_chi2['Chi2 Score'] >= chi2_threshold]
    significant_column_names = significant_features['Feature'].tolist()
    selected_features = data1[significant_column_names]
    A = selected_features  
    b = data1['Attack Type'] 
    A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.4, random_state=42)
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(A_train, b_train)
    b_pred = rf_classifier.predict(A_test)
    decoded_labels = label_encoders['Attack Type'].inverse_transform(b_pred)
    json_data = json.dumps(decoded_labels.tolist())
    return (json_data)


app = Flask(_name_) 
api = Api(app) 

value=mlnetwork()
attacktypevalue=attacktype()


class attack_or_normal(Resource): 
	def get(self): 
		data={ 
			'predicted values':value,
		} 
		return data 
api.add_resource(attack_or_normal,'/attack') 


class attacktype(Resource): 
	def get(self): 
		data={ 
			'predicted values':attacktypevalue,
		} 
		return data 

api.add_resource(attacktype,'/attacktype') 


if _name=='main_': 
	app.run(debug=True)
*