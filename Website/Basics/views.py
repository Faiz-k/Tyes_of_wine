from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def home(request):
    if request.method == "POST":
        try:
            fixed_acidity = request.POST.get('fixed_acidity')
            volatile_acidity = request.POST.get('volatile_acidity')
            citric_acid = request.POST.get('citric_acid')
            residual_sugar = request.POST.get('residual_sugar')
            chlorides = request.POST.get('chlorides')
            free_sulfur_dioxide = request.POST.get('free_sulfur_dioxide')
            total_sulfur_dioxide = request.POST.get('total_sulfur_dioxide')
            density = request.POST.get('density')
            pH = request.POST.get('pH')
            sulphates = request.POST.get('sulphates')
            alcohol = request.POST.get('alcohol')
        except ValueError:
            return render(request, 'home.html', context={'error': 'Invalid input. Please enter numeric values.'})

        # Load and preprocess the dataset (similar to your code)
        # ...
        data = pd.read_csv('C:\\Users\\mf879\\OneDrive\\Desktop\\2023_24projects\\2023_projects\\43_winequalityscoreClassification\\WineQT.csv')

# Select features (independent variables) and target (dependent variable)
        X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
        y = data['quality']

# Split the dataset into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
        nb_classifier = GaussianNB()

# Fit the classifier to the training data
        nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
        y_pred = nb_classifier.predict(X_test)

#print(y_pred)
        y=nb_classifier.predict([[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]])
        input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                                      columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                               'pH', 'sulphates', 'alcohol'])

            # Make predictions using the model
        predicted_quality = nb_classifier.predict(input_data)
        return render(request, "home.html", context={'predicted_type':predicted_quality[0] })

    return render(request, 'home.html')


