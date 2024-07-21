from django.shortcuts import render, redirect
from django.core.mail import send_mail
from django.contrib import messages
from .models import Slider, Service, Doctor, Faq, Gallery
from django.views.generic import ListView, DetailView, TemplateView

import pandas as pd
import os
import numpy as np
import pickle

from joblib import load
import pandas as pd   
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import datasets, linear_model, metrics

from django.shortcuts import render
from django.http import HttpResponse
from .forms import PredictionForm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


import random
import os
import numpy as np


from django.shortcuts import render
from django.http import HttpResponse
from .model_utils import load_model_and_predict
from django.http import HttpResponse
from django.views import View
from .models import Product

from nltk.chat.util import Chat, reflections

# from . import heart


class HomeView(ListView):
    template_name = 'hospital/index.html'
    queryset = Service.objects.all()
    context_object_name = 'services'

    def get_context_data(self, **kwargs):
        context = super().get_context_data()
        context['sliders'] = Slider.objects.all()
        context['experts'] = Doctor.objects.all()
        return context


class ServiceListView(ListView):
    queryset = Service.objects.all()
    template_name = "hospital/services.html"


class ServiceDetailView(DetailView):
    queryset = Service.objects.all()
    template_name = "hospital/service_details.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["services"] = Service.objects.all()
        return context


class DoctorListView(ListView):
    template_name = 'hospital/team.html'
    queryset = Doctor.objects.all()
    paginate_by = 8


class DoctorDetailView(DetailView):
    template_name = 'hospital/team-details.html'
    queryset = Doctor.objects.all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["doctors"] = Doctor.objects.all()
        return context


class FaqListView(ListView):
    template_name = 'hospital/faqs.html'
    queryset = Faq.objects.all()


class GalleryListView(ListView):
    template_name = 'hospital/gallery.html'
    queryset = Gallery.objects.all()
    paginate_by = 9


class ContactView(TemplateView):
    template_name = "hospital/contact.html"

    def post(self, request, *args, **kwargs):
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        if subject == '':
            subject = "Heartcare Contact"

        if name and message and email and phone:
            send_mail(
                subject+"-"+phone,
                message,
                email,
                ['expelmahmud@gmail.com'],
                fail_silently=False,
            )
            messages.success(request, " Email hasbeen sent successfully...")

        return redirect('contact')


def pred1(request):
    return render(request, 'hospital/pred1.html')
def analyze(request):
    heart = pd.read_csv("dataset/heartdisease.csv")
    from sklearn.model_selection import train_test_split

    heart.drop_duplicates()

   
    attr=["age","sex","trestbps","chol","fbs","restecg","thalach", "thal","disease"]
    if heart.isna().any().sum() !=0:
        for i in range(len(attr)):
            heart[attr[i]].fillna(0.0, inplace = True)

    
    heart.columns = heart.columns.str.replace(' ', '') 

   
    features = heart[['age','sex','trestbps','chol','fbs','restecg','thalach', 'thal']]

    
    target = heart['disease']

    x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)
    
    
    
    RF = RandomForestClassifier(n_estimators=20, random_state=0)

   
    RF.fit(x_train,y_train)
   
    age = request.POST.get('age', 'default')
    sex = request.POST.get('sex', 'default')
    trestbps = request.POST.get('trestbps', 'default')
    chol = request.POST.get('chol', 'default')
    fbs = request.POST.get('fbs', 'default')
    restecg =request.POST.get('restecg', 'default')
    thalach = request.POST.get('thalach', 'default')
    thal = request.POST.get('thal', 'default')



   #  Implementation

    userInput = [age, sex, trestbps, chol, fbs, restecg, thalach, thal]
   
    result1 = RF.predict([userInput])[0]

   
    params = {'purpose':'Predicted heart disease from Random forest classifier is: ', 'analyzed_text': str(result1).upper()}
    return render(request, 'hospital/analyze.html', params)   


def analyze(request):
    heart2 = pd.read_csv("dataset/heartdisease.csv")

    heart2.drop_duplicates()

    attr=["age","sex","trestbps","chol","fbs","restecg","thalach", "thal","disease"]
    if heart2.isna().any().sum() !=0:
        for i in range(len(attr)):
            heart2[attr[i]].fillna(0.0, inplace=True)

    heart2.columns = heart2.columns.str.replace(' ', '')

    feature = heart2[['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
    goal = heart2['disease']

    x_train, x_test, y_train, y_test = train_test_split(feature, goal, test_size=0.2, random_state=2)

    # Create KNeighborsClassifier instance
    KNN = KNeighborsClassifier(n_neighbors=5)

    KNN.fit(x_train, y_train)

    age = float(request.POST.get('age', '0.0'))
    sex = float(request.POST.get('sex', '0.0'))
    trestbps = float(request.POST.get('trestbps', '0.0'))
    chol = float(request.POST.get('chol', '0.0'))
    fbs = float(request.POST.get('fbs', '0.0'))
    restecg = float(request.POST.get('restecg', '0.0'))
    thalach = float(request.POST.get('thalach', '0.0'))
    thal = float(request.POST.get('thal', '0.0'))

    # Implementation

    Input = [[age, sex, trestbps, chol, fbs, restecg, thalach, thal]]

    result2 = KNN.predict(Input)[0]

    params = {'purpose': 'Predicted heartdiseae from KNeighbours Classifier is:', 'analyzed2': str(result2).upper()}
    return render(request, 'hospital/analyze.html', params)

from sklearn.svm import SVC  # Import Support Vector Classifier

from sklearn.svm import SVC  # Import Support Vector Classifier

def analyze(request):
    heart2 = pd.read_csv("dataset/heartdisease.csv")

    heart2.drop_duplicates()

    attr = ["age", "sex", "trestbps", "chol", "fbs", "restecg", "thalach", "thal", "disease"]
    if heart2.isna().any().sum() != 0:
        for i in range(len(attr)):
            heart2[attr[i]].fillna(0.0, inplace=True)

    heart2.columns = heart2.columns.str.replace(' ', '')

    feature = heart2[['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
    goal = heart2['disease']

    x_train, x_test, y_train, y_test = train_test_split(feature, goal, test_size=0.2, random_state=2)

    # Create classifiers
    KNN = KNeighborsClassifier(n_neighbors=5)
    SVM = SVC(kernel='linear', probability=True)  # Use linear kernel for SVM

    KNN.fit(x_train, y_train)
    SVM.fit(x_train, y_train)

    age = float(request.POST.get('age', '0.0'))
    sex = float(request.POST.get('sex', '0.0'))
    trestbps = float(request.POST.get('trestbps', '0.0'))
    chol = float(request.POST.get('chol', '0.0'))
    fbs = float(request.POST.get('fbs', '0.0'))
    restecg = float(request.POST.get('restecg', '0.0'))
    thalach = float(request.POST.get('thalach', '0.0'))
    thal = float(request.POST.get('thal', '0.0'))

    # User input
    input_data = [[age, sex, trestbps, chol, fbs, restecg, thalach, thal]]

    # Predict using KNN and SVM classifiers
    result_knn = KNN.predict(input_data)[0]
    result_svm = SVM.predict(input_data)[0]

    params = {
        'result_knn': str(result_knn[0]).upper(),
        'analyzed_svm': str(result_svm[0]).upper()
    }
    return render(request, 'hospital/analyze.html', params)


# def analyze(request):
#     heart = pd.read_csv("dataset/heartdisease.csv")

#     heart.drop_duplicates()

#     attr = ["age", "sex", "trestbps", "chol", "fbs", "restecg", "thalach", "thal", "disease"]
#     if heart.isna().any().sum() != 0:
#         for i in range(len(attr)):
#             heart[attr[i]].fillna(0.0, inplace=True)

#     heart.columns = heart.columns.str.replace(' ', '')

#     features = heart[['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'thal']]
#     target = heart['disease']

#     x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

#     # Create XGBClassifier instance
#     xgb_model = XGBClassifier()

#     xgb_model.fit(x_train, y_train)

#     age = float(request.POST.get('age', '0.0'))
#     sex = float(request.POST.get('sex', '0.0'))
#     trestbps = float(request.POST.get('trestbps', '0.0'))
#     chol = float(request.POST.get('chol', '0.0'))
#     fbs = float(request.POST.get('fbs', '0.0'))
#     restecg = float(request.POST.get('restecg', '0.0'))
#     thalach = float(request.POST.get('thalach', '0.0'))
#     thal = float(request.POST.get('thal', '0.0'))

#     # Implementation

#     userInput = [[age, sex, trestbps, chol, fbs, restecg, thalach, thal]]

#     result1 = xgb_model.predict(userInput)[0]  # Use XGBClassifier for prediction

#     params = {'purpose': 'Predicted heart: ', 'analyzed_text': str(result1).upper()}
#     return render(request, 'hospital/analyze.html', params)


def uploaded(request):
    if request.method == 'POST':
        f = request.FILES['file']
        if f is None:
            return HttpResponse('Please upload an image file.')
        path = 'media/data/' + f.name
        with open(path, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        
        disease, accuracy, img_filename = load_model_and_predict(path)
        
        return render(request, 'hospital/uploaded.html', {

            'predictions': disease,
            'acc': accuracy * 100,
            'img_file': img_filename,
        })
    else:
        return HttpResponse(status=400)

def skin(request):
    return render(request, 'hospital/skin.html')



def ecom(request):
    products=Product.objects.all()
    return render(request,'hospital/ecom.html',{'products':products})

def base(request):
    return render(request,'hospital/base.html')
    
def FAQs_view(request):
    return render(request, 'hospital/FAQs.html')

def registration(request):
    return render(request, 'appointment/registration.html')

def login(request):
    return render(request, 'appointment/login1.html')


def chatbot_view(request):
    chat_history = []
    
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        chat_history.append({"role": "user", "content": user_input})
        
        chat_pairs = [
            (r'(.*)appointment(.*)', ['Sure, when would you like to schedule your appointment?', 'Please provide your preferred date and time.']),
            (r'(.*)prescription(.*)', ['I can help with that. Please provide the name of the medication you need a prescription for.']),
            (r'(.*)prescribe for cold(.*)', ['Sure you can take Paracetamol as the remedy for cold.']),
            (r'(.*)Hi(.*)', ['Hello how can i help you!.']),
            (r'(.*)Who are you(.*)', ['Who the hell are you to ask me this question.']),
            (r'(.*)fever(.*)', ['Sure you can  take Arsenic Album 300.']),
            (r'(.*)prescribe for jont pain(.*)', ['Sure you can  take RRA']),
            (r'(.*)pimples(.*)', ['The reason of pimples are mainly: Your unhealthy lifestyle so hit the gym']),
            
            (r'(.*)', ["I'm sorry, I don't have a response for that.", "Could you please rephrase your question?"])
        ]
        
        chatbot = Chat(chat_pairs, reflections)
        bot_response = chatbot.respond(user_input)
        chat_history.append({"role": "bot", "content": bot_response})
        
    return render(request, 'hospital/chatbot.html', {'chat_history': chat_history})

# def predict_heart_disease(request):
#     # Your logic to handle predictions here
#     prediction_result = heart.predict(...)  # Call the appropriate function from the heart_predictor module
#     context = {
#         'prediction_result': prediction_result,
#     }
#     return render(request, 'hospital/analyze.html', context)


# def analyze(request):
#     heart = pd.read_csv("dataset/heartdisease.csv")

#     # ... your preprocessing code ...
#     from sklearn.model_selection import train_test_split

#     heart.drop_duplicates()

   
#     attr=["age","sex","trestbps","chol","fbs","restecg","thalach", "thal","disease"]
#     if heart.isna().any().sum() !=0:
#         for i in range(len(attr)):
#             heart[attr[i]].fillna(0.0, inplace = True)

    
#     heart.columns = heart.columns.str.replace(' ', '') 

   
#     features = heart[['age','sex','trestbps','chol','fbs','restecg','thalach', 'thal']]

    
#     target = heart['disease']

#     x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)

#     # Create classifiers
#     RF = RandomForestClassifier(n_estimators=20, random_state=0)
#     KNN = KNeighborsClassifier(n_neighbors=5)
#     SVM = SVC(probability=True)

#     # Fit models
#     RF.fit(x_train, y_train)
#     KNN.fit(x_train, y_train)
#     SVM.fit(x_train, y_train)

#     # Prepare user input
#     # ...
#     age = float(request.POST.get('age', '0.0'))
#     sex = float(request.POST.get('sex', '0.0'))
#     trestbps = float(request.POST.get('trestbps', '0.0'))
#     chol = float(request.POST.get('chol', '0.0'))
#     fbs = float(request.POST.get('fbs', '0.0'))
#     restecg = float(request.POST.get('restecg', '0.0'))
#     thalach = float(request.POST.get('thalach', '0.0'))
#     thal = float(request.POST.get('thal', '0.0'))

#     userInput = [age, sex, trestbps, chol, fbs, restecg, thalach, thal]

#     result1 = RF.predict([userInput])[0]
#     result2 = KNN.predict([userInput])[0]
#     result_svm = SVM.predict([userInput])[0]

#     # Evaluate algorithms using cross-validation
#     algorithms = [RF, KNN]
#     algorithm_names = ['Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine']
#     scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

#     results = []

#     for clf, name in zip(algorithms, algorithm_names):
#         scores = cross_validate(clf, features, target, cv=5, scoring=scoring_metrics)
#         avg_scores = [np.mean(scores[f'test_{metric}']) for metric in scoring_metrics]
#         print(f'{name} - Avg Scores: {avg_scores}')
#         results.append((name, avg_scores))

#     # Choose the algorithm with the best accuracy
#     best_index = np.argmax([result[1][0] for result in results])
#     best_algorithm, best_scores = results[best_index]

#     best_clf = algorithms[best_index]
#     best_result = best_clf.predict([userInput])[0]

#     # Prepare template context
#     context = {
#         'algorithm_results': results,
#         'best_algorithm': best_algorithm,
#         'best_scores': best_scores,
#         'best_result': str(best_result).upper(),
#         'analyzed_text': str(result1).upper(),
#         'analyzed2': str(result2).upper(),
#         'analyzed_svm': str(result_svm).upper(),
        

#     }

#     return render(request, 'hospital/analyze.html', context)




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter
from django.shortcuts import render

def analyze(request):
    
    heart = pd.read_csv("dataset/heartdisease.csv")

   
    from sklearn.model_selection import train_test_split

    heart.drop_duplicates()

   
    attr=["age","sex","trestbps","chol","fbs","restecg","thalach", "thal","disease"]
    if heart.isna().any().sum() !=0:
        for i in range(len(attr)):
            heart[attr[i]].fillna(0.0, inplace = True)

    
    heart.columns = heart.columns.str.replace(' ', '') 

   
    features = heart[['age','sex','trestbps','chol','fbs','restecg','thalach', 'thal']]

    
    target = heart['disease']

    x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)

   
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    KNN = KNeighborsClassifier(n_neighbors=5)
    SVM = SVC(probability=True)  
    LR = LogisticRegression(max_iter=1000)

  
    NB = GaussianNB()

   
    RF.fit(x_train, y_train)
    KNN.fit(x_train, y_train)
    SVM.fit(x_train, y_train)

    LR.fit(x_train, y_train)
    NB.fit(x_train, y_train)

    y_pred_svm = SVM.predict(x_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    age = float(request.POST.get('age', '0.0'))
    sex = float(request.POST.get('sex', '0.0'))
    trestbps = float(request.POST.get('trestbps', '0.0'))
    chol = float(request.POST.get('chol', '0.0'))
    fbs = float(request.POST.get('fbs', '0.0'))
    restecg = float(request.POST.get('restecg', '0.0'))
    thalach = float(request.POST.get('thalach', '0.0'))
    thal = float(request.POST.get('thal', '0.0'))

    userInput = [age, sex, trestbps, chol, fbs, restecg, thalach, thal]

   
    result_rf = RF.predict([userInput])[0]
    result_svm = SVM.predict([userInput])[0]
    result2 = KNN.predict([userInput])[0]

    result_lr = LR.predict([userInput])[0]
    result_nb = NB.predict([userInput])[0]
    

    
    predictions = [result_rf, result_svm]
    combined_prediction = Counter(predictions).most_common(1)[0][0]

    predictions = [result_lr, result_nb]
    combined_prediction2 = Counter(predictions).most_common(1)[0][0]

    predictions = [result_rf, result_lr, result_nb, result2, result_svm]
    combined_prediction3 = Counter(predictions).most_common(1)[0][0]

    # combined_predictionsA1 = [result_rf, result_svm]  
    # actual_labels = y_test  

    
    algorithms = [RF, KNN, SVM]
    algorithm_names = ['Random Forest', 'K-Nearest Neighbors', 'Support Vector Machine']

    algorithms = [LR, NB]
    algorithm_names = ['XGBoost',  'Naive Bayes']

    algorithms = [RF, LR, NB, KNN, SVM]
    algorithm_names = ['Random Forest', 'Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine']

    
    
    predictions = [result_rf, result_svm, result2, result_lr, result_nb]


    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # accuracy = accuracy_score(actual_labels, combined_predictionsA1)

   

    results = []

    for clf, name in zip(algorithms, algorithm_names):
        scores = cross_validate(clf, features, target, cv=5, scoring=scoring_metrics)
        avg_scores = [np.mean(scores[f'test_{metric}']) for metric in scoring_metrics]
        print(f'{name} - Avg Scores: {avg_scores}')
        results.append((name, avg_scores))

        
   
    best_index = np.argmax([result[1][0] for result in results])
    best_algorithm, best_scores = results[best_index]

    best_clf = algorithms[best_index]
    best_result = best_clf.predict([userInput])[0]

   

    # label_encoder = LabelEncoder()
    # heart['disease'] = label_encoder.fit_transform(heart['disease'])

    
    context = {
        'algorithm_results': results,
        'best_algorithm': best_algorithm,
        'best_scores': best_scores,
        'best_result': str(best_result).upper(),
        'analyzed_text_rf': str(result_rf).upper(),
        'analyzed_text_svm': str(result_svm).upper(),
        'combined_prediction': str(combined_prediction).upper(),
        'analyzed2': str(result2).upper(),

        'analyzed_text_lr': str(result_lr).upper(),
        'analyzed_text_nb': str(result_nb).upper(),
        'combined_prediction2': str(combined_prediction2).upper(),
        'combined_prediction3': str(combined_prediction3).upper(),
        # 'accuracy': accuracy,


        

        
    }

    return render(request, 'hospital/analyze.html', context)


