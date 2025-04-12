import imaplib
import os
from dotenv import load_dotenv

# Load the environment variables from .env
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("IMAP_SERVER")

print("EMAIL_USER:", repr(EMAIL_USER))
print("EMAIL_PASS:", repr(EMAIL_PASS))
print("IMAP_SERVER:", repr(IMAP_SERVER))


try:
    print("Connecting to server...")
    # mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    # mail.login(EMAIL_USER, EMAIL_PASS)
    mail = imaplib.IMAP4(IMAP_SERVER, 993)  # Use port 143
    mail.starttls()  # Upgrade to a secure connection
    mail.login(EMAIL_USER, EMAIL_PASS)

    print("✅ Login successful!")
    mail.logout()
except imaplib.IMAP4.error as e:
    print(f"❌ IMAP error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
