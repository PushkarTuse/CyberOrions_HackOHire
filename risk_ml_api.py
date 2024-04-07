
from flask import Flask 
from flask_restful import Api, Resource 
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
df = pd.read_csv(r"C:\Users\nitya mohta\Downloads\cybersec_database.csv")

def calculate_risk_overall(row):
    risk_factors = [0] * 10
    policy_violation_type = row['Policy Violation Type']
    threat_actor_tactic = row['Threat Actor Tactic']
    threat_actor_procedure = row['Threat Actor Procedure']
    user_behavior_pattern = row['User Behavior Pattern']
    asset_risk_profile = row['Asset Risk Profile']
    patches_applied = row['Patches Applied']
    outstanding_patches = row['Outstanding Patches']
    successful_patch_deployments = row['Successful Patch Deployments']
    network_traffic_pattern = row['Network Traffic Pattern']
    security_awareness_training_completion_rate = row['Security Awareness Training Completion Rate']
    third_party_security_incident = row['Third-party Security Incident']
    security_awareness_training_completion_rate = pd.to_numeric(security_awareness_training_completion_rate, errors='coerce')
    policy_violation_type_num = 1 if policy_violation_type == 'High' else 0
    threat_actor_tactic_num = 1 if threat_actor_tactic == 'Malware Distribution' else 0
    threat_actor_procedure_num = 1 if threat_actor_procedure == 'Phishing' else 0
    user_behavior_pattern_num = 1 if user_behavior_pattern in ['Late Night Logins', 'Unusual Access Patterns'] else 0
    asset_risk_profile_num = 1 if asset_risk_profile == 'High' else 0
    network_traffic_pattern_num = 1 if network_traffic_pattern == 'High' else 0
    third_party_security_incident_num = 1 if third_party_security_incident == 'Yes' else 0

    # Calculate risk factors using mathematical functions
    policy_violation_type_risk_factor = np.sin(policy_violation_type_num * np.pi / 2)
    threat_actor_tactic_risk_factor = np.tanh(threat_actor_tactic_num)
    threat_actor_procedure_risk_factor = 1 - np.exp(-threat_actor_procedure_num)
    user_behavior_pattern_risk_factor = np.arctan(user_behavior_pattern_num * np.pi / 4) / (np.pi / 2)
    asset_risk_profile_risk_factor = asset_risk_profile_num ** 2
    patches_applied_risk_factor = np.exp(-patches_applied / 20) if patches_applied > 0 else 0  # Exponential decay function
    outstanding_patches_risk_factor = np.log(outstanding_patches + 1) / np.log(21) if outstanding_patches > 0 else 0  # Logarithmic function
    successful_patch_deployments_risk_factor = 1 - (successful_patch_deployments / 20) if successful_patch_deployments > 0 else 0  # Linear function
    network_traffic_pattern_risk_factor = np.cos(network_traffic_pattern_num * np.pi / 2)
    security_awareness_training_completion_rate_risk_factor = 1 - (security_awareness_training_completion_rate / 100) if security_awareness_training_completion_rate > 0 else 0  # Linear function
    third_party_security_incident_risk_factor = 1 - np.exp(-third_party_security_incident_num)

    # Update the risk_factors list with the calculated risk factors
    risk_factors[0] = policy_violation_type_risk_factor
    risk_factors[1] = threat_actor_tactic_risk_factor
    risk_factors[2] = threat_actor_procedure_risk_factor
    risk_factors[3] = user_behavior_pattern_risk_factor
    risk_factors[4] = asset_risk_profile_risk_factor
    risk_factors[5] = patches_applied_risk_factor
    risk_factors[6] = outstanding_patches_risk_factor
    risk_factors[7] = successful_patch_deployments_risk_factor
    risk_factors[8] = network_traffic_pattern_risk_factor
    risk_factors[9] = security_awareness_training_completion_rate_risk_factor

    # Check for NaN or inf values and replace them with 0
    risk_factors = [0 if np.isnan(factor) or np.isinf(factor) else factor for factor in risk_factors]
    
    # Calculate the Pearson correlation between risk factors
    risk_factor_corr = [pearsonr(risk_factors, risk_factors)[0]] * len(risk_factors)

    overall_risk = sum(risk_factor * corr for risk_factor, corr in zip(risk_factors, risk_factor_corr))
    return overall_risk

def risk():
    # Apply the risk calculation function to the DataFrame and store the results in a new column
    df['Overall Risk'] = df.apply(calculate_risk_overall, axis=1)

    # Group by 'Location' and calculate the sum of 'Overall Risk' for each location
    overall_risk_by_location = df.groupby('Location')['Overall Risk'].sum()

    # Print the sum of overall risk ratings for each location
    json_string=overall_risk_by_location.to_json()
    return(json_string)

app = Flask(__name__) 
api = Api(app) 


value=risk()
class risk(Resource): 
	def get(self): 
		data={ 
			'predicted values':value,
		} 
		return data 
api.add_resource(risk,'/risk') 


if __name__=='__main__': 
	app.run(debug=True)
