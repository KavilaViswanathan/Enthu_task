from model import model, vectorizer  
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    user_input = input("Enter a review to analyze sentiment: ")
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")
