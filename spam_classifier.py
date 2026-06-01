import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and target
X = data['message']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert text to vectors
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vector, y_train)

# Evaluate model
predictions = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Test custom email
print("\n===== Spam Email Detector =====")

email = input("Enter Email Message:\n")

email_vector = vectorizer.transform([email])

result = model.predict(email_vector)

if result[0] == 1:
    print("\nSPAM EMAIL")
else:
    print("\nNOT SPAM")