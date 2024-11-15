# movemate/test_email_simple.py
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_smtp_connection(max_retries=3, timeout=30):
    # Email settings
    smtp_server = "mail.privateemail.com"
    port = 587
    sender_email = "sales@movemate.me"
    password = os.getenv('EMAIL_HOST_PASSWORD')
    receiver_email = "ismetsemedov@live.ru"

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Test Email from MoveMate"

    # Add body
    body = "This is a test email from MoveMate using direct SMTP."
    message.attach(MIMEText(body, "plain"))

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            
            # Create SMTP session with timeout
            print("Creating SMTP session...")
            server = smtplib.SMTP(smtp_server, port, timeout=timeout)
            
            print("Starting TLS...")
            server.starttls()
            
            print("Logging in...")
            server.login(sender_email, password)
            
            print("Sending email...")
            text = message.as_string()
            server.sendmail(sender_email, receiver_email, text)
            
            print("Email sent successfully!")
            server.quit()
            return True
            
        except smtplib.SMTPServerDisconnected as e:
            print(f"Server disconnected: {str(e)}")
        except smtplib.SMTPAuthenticationError as e:
            print(f"Authentication failed: {str(e)}")
            return False  # No point in retrying auth failures
        except TimeoutError as e:
            print(f"Timeout error: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                print("Closing server connection...")
                server.quit()
            except:
                pass
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5  # Progressive delay: 5s, 10s, 15s
            print(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    
    return False

if __name__ == "__main__":
    if test_smtp_connection():
        print("\nTest completed successfully!")
    else:
        print("\nTest failed after all retries.")