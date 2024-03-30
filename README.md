

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

# If want to build a normal sms classification . Do the following steps

To build the project and load the mail data in a Jupyter Notebook, follow these steps:

1. **Install Required Libraries**: First, make sure you have the necessary libraries installed. In this case, you'll need `numpy`, `pandas`, and `scikit-learn`.

   ```bash
   pip install numpy pandas scikit-learn
   ```

2. **Set Up Jupyter Notebook**: If you haven't already installed Jupyter Notebook, you can do so using pip:

   ```bash
   pip install jupyterlab
   ```

   Then, start Jupyter Notebook by running:

   ```bash
   jupyter notebook
   ```

3. **Create a New Notebook**: In the Jupyter Notebook interface, click on "New" and select "Python 3" to create a new Python notebook.

4. **Import Libraries**: In the first cell of your notebook, import the required libraries:

   ```python
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

5. **Load Mail Data**: Now, you can load your mail data into a pandas DataFrame. If your data is in a CSV file named `mail_data.csv`, you can use the following code:

   ```python
   raw_mail_data = pd.read_csv('mail_data.csv')
   ```

   Adjust the file path as necessary to point to your CSV file.

6. **Explore Data**: It's a good practice to take a quick look at your data to understand its structure. You can use pandas methods like `.head()` to display the first few rows of the DataFrame:

   ```python
   raw_mail_data.head()
   ```

7. **Data Preprocessing**: Perform any necessary preprocessing steps such as handling missing values, encoding categorical variables, or cleaning the text data.

8. **Split Data**: Split your data into training and testing sets using `train_test_split()`:

   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
   ```

9. **Feature Extraction**: Convert text data into numerical features using TF-IDF vectorization:

   ```python
   feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
   X_train_features = feature_extraction.fit_transform(X_train)
   X_test_features = feature_extraction.transform(X_test)
   ```

10. **Train Model**: Initialize and train your machine learning model (e.g., Logistic Regression):

    ```python
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    ```

11. **Evaluate Model**: Evaluate the performance of your model on the test data:

    ```python
    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    print('Accuracy on test data : ', accuracy_on_test_data)
    ```

12. **Use the Model**: Finally, you can use your trained model to make predictions on new data.

That's it! You've now built a project in Jupyter Notebook to load mail data, preprocess it, train a machine learning model, and evaluate its performance.
