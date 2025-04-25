import pandas as pd
from twilio.rest import Client
import os
import geocoder
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

g = geocoder.ip('me')
CAMERA_LOCATION = g.json['address']+f'. [Lat: {g.lat}, Lng:{g.lng}]'

def send_email(email, name, updated_penalty):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    
    sender_email = "griettrafficdepartment@gmail.com"  
    sender_password = "dlje syhc ltov xccu"  
    receiver_email = f"{email}"
    
    print(f"Message sent to email linked to registration: {receiver_email}")
    subject = "Penalty Email From GRIET-Traffic Department"
    body = f'{name}, you were caught riding without helmet near {CAMERA_LOCATION}, and were fined Rupees 100.Your total pending is {updated_penalty}.Please visit https://echallan.tspolice.gov.in/publicview/ to pay your due challan. If you are caught riding again without proper legal way, you will be severely penalized.Pay Your fine using this demo link:http://127.0.0.1:5000/payment_login'
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        server.login(sender_email, sender_password)
        
        server.sendmail(sender_email, receiver_email, message.as_string())
        
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error occurred while sending email: {e}")
    finally:
        server.quit()

def load_database():
    return pd.read_csv('data\database.csv')

def fetch_details(number_plate):
    database = load_database()
    record = database[database['Registration'] == number_plate]
    if record.empty:
        print(f"No record found for number plate: {number_plate}")
        return None
    
    mobile_number = record.iloc[0]['Mobile']
    email = record.iloc[0]['Email']
    name = record.iloc[0]['Name']
    penalty = float(record.iloc[0]['Challan'])
    
    updated_penalty = penalty + 100
    
    database.loc[database['Registration'] == number_plate, 'Challan'] = updated_penalty
    database.to_csv('data\database.csv', index=False)
        
    return mobile_number, updated_penalty, email, name

def send_sms(mobile_number, updated_penalty, number_plate, name):
    try:
        client = Client('ACa338eaefda6b134dbbf0cc5e9b8fca7e', '8dbfe23c02ddacb0732026b68e766b5d')
        
        message_body = f'{name}, You were caught riding without helmet near {CAMERA_LOCATION}, and were fined Rupees 100. Your total pending is {updated_penalty}. Please visit https://echallan.tspolice.gov.in/publicview/ to pay your due challan. If you are caught riding again without proper gear, you will be severely penalized.Pay Your fine using this demo link:http://127.0.0.1:5000/payment_login'
        
        message = client.messages.create(
            body=message_body,
            from_='+12525883621',
            to=f'+{mobile_number}'
        )
        
        print(f"Message sent to mobile number linked to registration: {number_plate}")
        return True
    except Exception as e:
        print(f"Error occurred while sending SMS: {e}")
        return False

def process_number_plate(number_plate):
    result = fetch_details(number_plate)
    
    if result:
        mobile_number, updated_penalty, email, name = result
        sms_sent = send_sms(mobile_number, updated_penalty, number_plate, name)
        send_email(email, name, updated_penalty)
        return True
    else:
        return False

def main():
    choice = input("Enter Text: ").strip()
    process_number_plate(choice)
  
if __name__ == "__main__":
    main()