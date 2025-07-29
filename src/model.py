# src/model.py

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_and_clean_data

# 1. Load and clean data
X, y = load_and_clean_data('dataset/spam.csv')


# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test_tfidf)

# 6. Show accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. BONUS: Predict on a custom message
def predict_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test the custom function
test_msg = "Congratulations! You've won a free ticket."
print("\nTest Message Prediction:", predict_message(test_msg))
# Test the custom function
test_msg = "Congratulations! You've won a free ticket."
print("\nTest Message Prediction:", predict_message(test_msg))

# 8. Save the classification report to a file
with open("results/classification_report.txt", "w") as f:
    f.write("Accuracy: {:.2f}%\n\n".format(accuracy_score(y_test, y_pred) * 100))
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
