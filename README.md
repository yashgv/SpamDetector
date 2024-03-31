# SpamDetector
Spam Detection with Streamlit Integration
This project implements a spam detection model using Naive Bayes and integrates it with a user-friendly Streamlit frontend. Users can enter email or SMS text, and the model predicts whether it's likely spam or not.

# Requirements:
Python 3
pandas
scikit-learn
streamlit

# Installation:
Install the required libraries: pip install pandas scikit-learn streamlit


# Code Structure:
<h4>Data Loading and Preprocessing:</h1>
Loads the spam dataset using pandas.
(Optional) Performs basic cleaning steps like removing unnecessary columns or handling missing values.
Renames columns for better readability.
Converts the 'label' column to numerical values (0 for ham, 1 for spam).

<h4>Model Training:</h4>
Splits the data into training and testing sets.
Trains a Naive Bayes model using MultinomialNB.
(Optional) Train an additional model like Random Forest for comparison (commented out in the example).

<h4>Streamlit Integration:</h4>
Creates a Streamlit app with a title ("Spam Detector").
Provides a text input field for users to enter email or SMS content.
Uses the check function (or similar logic) to predict spam likelihood.
Displays the prediction as a success ("Not Spam") or error ("Spam") message.
