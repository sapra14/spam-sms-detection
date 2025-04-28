import pickle

# Load trained model and vectorizer
with open('outputs/spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('outputs/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Predict function
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test examples
if __name__ == "__main__":
    user_text = input("Enter a message: ")
    result = predict_spam(user_text)
    print(f"The message is classified as: {result}")
