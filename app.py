import os
import imaplib
import email
import joblib
import re
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Trello API credentials
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_SECRET_KEY = os.getenv("TRELLO_SECRET_KEY")
TRELLO_LIST_ID = "67f92c657ad453b5b2d74825"
# Email credentials
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("IMAP_SERVER")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# Ticket API URL
TICKET_API_URL = os.getenv("TICKET_API_URL")

# Service group mapping
SERVICE_GROUP_MAPPING = {
    "login issue": "Authentication Team",
    "payment failure": "Billing Team",
    "bug": "Engineering Team",
    "feature request": "Product Team",
    "general": "Support Team"
}

# Load CSV file
df = pd.read_csv("customer_issues.csv")
df.dropna(inplace=True)

# Extract features and labels
X = df["Issue Description"]
y = df["Issue Type"]

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42)

# Train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the updated model and vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, labels=range(len(label_encoder.classes_))
))

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
print("Vectorizer loaded. ")
model = joblib.load('model.pkl')
print("Model loaded.")
# Load the label encoder

def clean_text(text):
    """Cleans the input text by removing special characters and converting to lowercase."""
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())

def classify(text):
    """Classifies the input text into a category using the trained model."""
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    encoded_label = model.predict(vect)[0]
    return label_encoder.inverse_transform([encoded_label])[0]  # Convert back to string label

# def create_trello_card(issue, category, group, customer_email):
#     """Creates a card in Trello."""
#     url = "https://api.trello.com/1/cards"
#     # url = "https://api.trello.com/1/cards"
#     query = {
#         'key': TRELLO_API_KEY,
#         'token': TRELLO_SECRET_KEY,
#         # 'idList':"https://api.trello.com/1/boards/SbvpVhIQ/lists?key=812bd10bece41a5db35dcfe1307370e4&token=ATTA94f73b884f09e72efe450172c62c6e6495980150a4e8b99a241d90bb7d347e4fD665BF01",
#         'idList':"67f9b701dee2ccf84316c9a0",
#         'name': issue if issue.strip() else "No issue provided",
#         'desc': f"Category: {category}\nAssigned Group: {group}\nCustomer Email: {customer_email}"
#     }
#     print("Trello Query:", query)
#     try:
#         response = requests.post(url, params=query)
#         if response.status_code == 200 or response.status_code == 201:
#             print("✅ Card created successfully:", response.json())
#         else:
#             print(f"❌ Failed to create card: {response.status_code} - {response.text}")
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Error creating card: {e}")

# def create_trello_card(issue, category, group, customer_email):
#     print('Creating Trello card...')
#     print(f"URL: https://api.trello.com/1/cards")
#     """Creates a card in Trello."""
#     url = "https://api.trello.com/1/cards"
#     # print("Creating Trello card...")
#     print(f"URL 2: {url}")
#     query = {
#         'key': TRELLO_API_KEY,
#         'token': TRELLO_SECRET_KEY,
#         'idList': "67f9b701dee2ccf84316c9a0",  # Make sure this is correct!
#         'name': issue if issue.strip() else "No issue provided",
#         'desc': f"Category: {category}\nAssigned Group: {group}\nCustomer Email: {customer_email}"
#     }

#     print("Trello Query:", query)

#     try:
#         response = requests.post(url, params=query)
#         if response.status_code in [200, 201]:
#             print("✅ Card created successfully:", response.json())
#         else:
#             print(f"❌ Failed to create card: {response.status_code} - {response.text}")
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Error creating card: {e}")

def create_trello_card(issue, category, group, customer_email):
    """Creates a card in Trello."""
    print('Creating Trello card...')
    print(f"URL: https://api.trello.com/1/cards")
    url = "https://api.trello.com/1/cards"
    print(f"URL 2: {url}")
    query = {
        'key': TRELLO_API_KEY,
        'token': TRELLO_SECRET_KEY,
        'idList': "67f92c657ad453b5b2d74825",  # Make sure this is correct!
        'name': issue if issue.strip() else "No issue provided",
        'desc': f"Category: {category}\nAssigned Group: {group}\nCustomer Email: {customer_email}"
    }

    print("Trello Query:", query)

    try:
        response = requests.post(url, params=query)
        if response.status_code in [200, 201]:
            print("✅ Card created successfully:", response.json())
        else:
            print(f"❌ Failed to create card: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating card: {e}")

def send_email(to_email, subject, body):
    """Sends an email with the ticket details."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def fetch_emails():
    """Fetches unread emails from the inbox."""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        if status != "OK":
            print("No unread emails found.")
            return []

        emails = []
        for num in messages[0].split():
            try:
                status, msg_data = mail.fetch(num, '(RFC822)')
                if status != "OK":
                    print(f"Failed to fetch email with ID {num}. Skipping...")
                    continue

                msg = email.message_from_bytes(msg_data[0][1])
                subject = msg["subject"]
                from_ = msg["from"]
                body = ""

                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                body += part.get_payload(decode=True).decode()
                            except (UnicodeDecodeError, AttributeError):
                                body += ""
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except (UnicodeDecodeError, AttributeError):
                        body = ""

                emails.append({"from": from_, "subject": subject, "body": body})
            except Exception as e:
                print(f"Error processing email with ID {num}: {e}")
        mail.logout()
        return emails
    except imaplib.IMAP4.error as e:
        print(f"IMAP error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def create_ticket(issue, category, group, customer_email):
    """Creates a ticket in the ticketing system."""
    ticket_data = {
        "subject": str(issue),
        "description": str(issue),
        "category": str(category),
        "assigned_group": str(group),
        "customer_email": str(customer_email)
    }
    params = {
        'key': TRELLO_API_KEY,
        'token': TRELLO_SECRET_KEY,
        'idList': "67f92c657ad453b5b2d74825",  # Make sure this is correct!
        'name': issue if issue.strip() else "No issue provided",
        # 'desc': f"Category: {category}\nAssigned Group: {group}\nCustomer Email: {customer_email}"
    }
    print("Creating ticket with data:", ticket_data)
    try:
        response = requests.post(TICKET_API_URL, json=ticket_data,params=params)
        if response.status_code == 201:
            print(f"✅ Ticket created successfully for issue: {issue}")
            send_email(customer_email, "Your Ticket Details", f"Subject: {issue}\nCategory: {category}\nAssigned Group: {group}")
        else:
            print(f"❌ Failed to create ticket: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating ticket in create Ticket(): {e}")

def process_emails():
    """Processes unread emails by classifying them and mapping them to the appropriate service group."""
    emails = fetch_emails()
    if not emails:
        print("No emails to process.")
        return

    for email_data in emails:
        try:
            issue = email_data["body"]
            if not issue.strip():
                issue = "No content available"  # Fallback for empty issues

            category = classify(issue)
            group = SERVICE_GROUP_MAPPING.get(category, SERVICE_GROUP_MAPPING["general"])
            customer_email = email_data["from"]

            print(f"Email from: {email_data['from']}")
            print(f"Subject: {email_data['subject']}")
            print(f"Issue: {issue}")
            print(f"Category: {category}")
            print(f"Assigned Group: {group}")
            print("-" * 50)

            # Create a ticket and send email
            create_ticket(issue, category, group, customer_email)

        except Exception as e:
            print(f"Error processing email: {e}")

if __name__ == "__main__":
    print("Processing unread emails...")
    process_emails()