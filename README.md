

# SMS Classifier 📱🔍

A SMS classifier is a machine learning model trained to categorize text messages, typically into spam or non-spam (ham) categories. This classification task involves processing large volumes of text data and extracting relevant features to distinguish between spam and legitimate messages.

## Implementation Overview 🧠📊

The provided code demonstrates the implementation of a SMS classifier using logistic regression and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. 

1. **Data Preprocessing**: The dataset containing SMS messages is loaded and preprocessed, including handling missing values and encoding the target categories.

2. **Feature Extraction**: Text data is transformed into numerical features using TF-IDF vectorization, which captures the importance of words in each message.

3. **Model Training**: A logistic regression model is trained on the training data, leveraging these features to learn patterns distinguishing between spam and ham messages.

4. **Model Evaluation**: The accuracy of the model is evaluated on both training and testing data.

5. **Prediction**: Finally, the model is used to predict the category of a new input message, demonstrating its practical application in classifying SMS messages as spam or ham.

## Usage

To use this code:

1. Ensure you have the necessary libraries installed (`numpy`, `pandas`, `scikit-learn`).
2. Provide your SMS dataset in CSV format.
3. Execute the provided Python script.
