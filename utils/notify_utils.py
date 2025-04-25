# utils/notify_utils.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client

def send_email(recipient, license_plate, location):
    """
    Sends an email notification.
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = "Helmet Violation Alert"
    message["From"] = "your.email@example.com"
    message["To"] = recipient
    body = (f"You were detected riding without a helmet near {location}. "
            f"License Plate: {license_plate}. "
            "A fine has been issued. Please check your details.")
    message.attach(MIMEText(body, "plain"))
    
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login("your.email@example.com", "your_email_password")
    server.sendmail("your.email@example.com", recipient, message.as_string())
    server.quit()

def send_sms(phone_number, license_plate, location):
    """
    Sends an SMS notification.
    """
    account_sid = "YOUR_TWILIO_ACCOUNT_SID"
    auth_token = "YOUR_TWILIO_AUTH_TOKEN"
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=(f"Helmet Violation Alert: Detected license plate {license_plate} near {location}. "
              "A fine has been issued."),
        from_="+1234567890",  # Your Twilio number
        to=phone_number
    )
    return message.sid
