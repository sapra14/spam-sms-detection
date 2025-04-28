# spam-sms-detection
Machine Learning model to classify SMS messages as spam or ham.

Project Objective
The objective of this project is to develop a machine learning model capable of accurately classifying SMS messages as Spam or Ham (non-spam).
This project uses text preprocessing, feature extraction (TF-IDF Vectorization), and a Logistic Regression classifier to detect spam messages.
It demonstrates the ability to work with real-world text data and build an end-to-end machine learning pipeline.

Project Structure

spam-sms-detection/
├── data/
│   └── spam.csv                # Dataset file (SMS messages and labels)
├── outputs/
│   ├── spam_classifier.pkl     # Saved trained model
│   ├── vectorizer.pkl          # Saved TF-IDF vectorizer
│   └── confusion_matrix.png    # Evaluation plot
├── train.py                    # Model training and evaluation script
├── predict.py                  # Script to predict custom user messages
├── README.md                   # Project documentation
├── requirements.txt            # List of required Python libraries
└── .gitignore                  # Ignored files and folders

IMPLEMENTATION
1.Importing Libraries:
Libraries like pandas, numpy, sklearn, matplotlib, and pickle are used for data manipulation, machine learning, model evaluation, and saving/loading the model.

2.Loading Data:
The dataset (spam.csv) is loaded into a DataFrame. It contains two columns: v1 (label: "ham" or "spam") and v2 (SMS text). The DataFrame is renamed to label and text for better clarity.

3.Preprocessing:
Labels are mapped to numerical values: 0 for "ham" and 1 for "spam".
X contains the text messages, and y contains the corresponding labels.

4.Splitting Data:
The data is split into training (80%) and testing (20%) sets using train_test_split.

5.Vectorization:
Text data is converted into numeric form using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which is applied on both the training and test datasets.

6.Model Training:
A Logistic Regression model is trained using the vectorized text data (X_train_vec) and the corresponding labels (y_train).

7.Evaluation:
The model’s performance is evaluated using the classification_report and a confusion matrix. The confusion matrix is visualized and saved as an image.

8.Saving the Model:
Both the trained model and the vectorizer are saved as .pkl files using pickle, allowing the model to be reloaded and used later.

9.Prediction Function:
The predict_spam function takes user input, vectorizes it, and predicts whether the text is "Spam" or "Ham".

10,User Input:
When the script is run, it prompts the user for a text message and then classifies it as either "Spam" or "Ham" using the trained model.


Key Features
1.TF-IDF Feature Extraction
Captures the importance of words across SMS messages for better model performance.

2.Logistic Regression Classifier
Lightweight and effective for binary classification tasks like spam detection.

3.Model Persistence
Both the model and vectorizer are saved using Pickle for easy reusability.

4.Custom Message Prediction
Users can input their own messages to check if they would be classified as spam.

5.Evaluation Metrics
Classification Report (Precision, Recall, F1-Score) and Confusion Matrix.

Model Evaluation
Accuracy: Over 95% on the test set.
Confusion Matrix: Shows True Positives, True Negatives, False Positives, False Negatives.
Classification Report: Includes Precision, Recall, and F1-Score for both classes (Spam and Ham).
A sample Confusion Matrix is saved as outputs/confusion_matrix.png.

Technologies Used
Python

Pandas

NumPy

Scikit-learn

Matplotlib

Future Enhancements
1.Integrate Deep Learning models like LSTM for improved detection accuracy.
2.Build a Web App using Flask to deploy the model for public use.
3.Expand the dataset with more recent SMS data for better generalization.


Conclusion
This project showcases a complete machine learning pipeline — from data preprocessing to model training and evaluation — to solve a real-world text classification problem.It also emphasizes modular code, reproducibility, and clear documentation, making it easy for others to understand and extend the work.

