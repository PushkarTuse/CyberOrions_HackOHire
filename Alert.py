import pandas as pd
import requests
import json
import base64

# Define your ClickSend API credentials
username = "user2314"
password = "FF40DD20-6685-085E-9706-E1F11505234D"

# Base URL for ClickSend API
base_url = "https://rest.clicksend.com/v3"

# Define your sender ID
sender_id = "user2314"

# Construct the authentication header
auth_header = {
    "Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
}

# Function to send SMS using ClickSend API
def send_sms(message, recipient):
    payload = {
        "messages": [
            {
                "source": "sdk",
                "from": sender_id,
                "body": message,
                "to": recipient
            }
        ]
    }
    endpoint = "/sms/send"
    response = requests.post(base_url + endpoint, headers=auth_header, json=payload)
    if response.status_code == 200:
        print(f"SMS sent successfully to {recipient}.")
    else:
        print(f"Failed to send SMS to {recipient}. Error:", response.text)

# Read the Excel file with phone numbers and weather alerts
excel_file_path = "BARCLAYS_MAIN.xlsx"  # Provide the correct path to your Excel file
df = pd.read_excel(excel_file_path)

# Iterate over each row
for index, row in df.iterrows():
    weather_alert = row["Weather_alerts"]
    phone_number = row["Column1"]

    # Send SMS with weather alert
    send_sms(weather_alert, phone_number)
    
