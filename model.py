
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

texts = [
    "The product is great and works perfectly",
    "Terrible experience, it broke in a day",
    "Very satisfied with the purchase",
    "I want a refund, not worth the money",
    "Excellent customer service",
    "Worst quality I’ve ever seen",
    "Super happy with the support team",
    "Not what I expected, very disappointed",
    "Highly recommend this to everyone",
    "Item was defective and unusable",
    "I love this product, it's fantastic!",
    "Complete waste of money, do not buy",
    "The service was slow and unhelpful",
    "Amazing product, highly recommend",
    "Very disappointed with the purchase"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = SVC()

model.fit(X_train,y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

new_text = ["The product is amazing"]
new_text_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_text_vectorized)
print(f"Prediction for '{new_text[0]}': {prediction[0]}")
