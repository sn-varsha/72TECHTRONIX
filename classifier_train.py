import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Can't login to my account", "Payment failed during checkout", "App crashes when I open settings",
    "Would love to see a dark mode", "I don't understand how to change my profile",
    "Forgot my password", "App not accepting credit card", "Bug in the notification system",
    "Add option to export data", "Need help navigating dashboard",
    "Account locked after multiple tries", "Charged twice for subscription",
    "Unexpected error occurred on dashboard", "Introduce referral program", "Where can I find the help section?",
    "Reset password not working", "Transaction failed", "Unable to upload profile picture",
    "Can you add a calendar feature?", "How to set up notifications?"
]
labels = [
    "login issue", "payment failure", "bug",
    "feature request", "general",
    "login issue", "payment failure", "bug",
    "feature request", "general",
    "login issue", "payment failure",
    "bug", "feature request", "general",
    "login issue", "payment failure", "bug",
    "feature request", "general"
]



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')
print("Model trained and saved.")