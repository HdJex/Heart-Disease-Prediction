# forms.py
from django import forms

class PredictionForm(forms.Form):
    # Define form fields based on the data you need for prediction
    field1 = forms.CharField(label='Field 1')
    field2 = forms.CharField(label='Field 2')
    # Add more fields as needed


