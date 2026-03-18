import smtplib
from email.message import EmailMessage
import sys

# Get the Chicken ID passed from C++
chicken_id = sys.argv[1]

# 1. Draft the Email
msg = EmailMessage()
msg.set_content(f"ALERT: Chicken with ID {chicken_id} has been flagged as UNHEALTHY by the TensorRT tracking system. Please check the pen.")
msg['Subject'] = f"Poultry Alert: Sick Chicken Detected (ID: {chicken_id})"
msg['From'] = "uduakdhona@gmail.com"
msg['To'] = "uduakdhona@gmail.com" # Send it to yourself

# 2. Connect to Gmail and Send
try:
    # Use standard Gmail SSL Port
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    # Put your email and the 16-letter App Password here
    server.login("uduakdhona@gmail.com", "vdix cvpc qszx hqbt") 
    server.send_message(msg)
    server.quit()
    print(f"Successfully sent email alert for ID {chicken_id}")
except Exception as e:
    print(f"Failed to send email: {e}")