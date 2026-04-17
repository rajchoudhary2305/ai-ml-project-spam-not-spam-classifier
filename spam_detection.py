# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Create simple dataset
messages = [
    "Win money now",
    "Call me later",
    "Free gift available",
    "Let's study together",
    "Earn cash easily",
    "Meeting at 5pm"
]

labels = [
    "spam",
    "ham",
    "spam",
    "ham",
    "spam",
    "ham"
]

# Step 2: Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Step 3: Train model
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Test with new message
test_message = ["Free money offer"]
test_data = vectorizer.transform(test_message)

# Step 5: Prediction
result = model.predict(test_data)

print("Message:", test_message[0])
print("Prediction:", result[0])
