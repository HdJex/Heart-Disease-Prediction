from django.urls import path
from . import views
from sre_constants import SUCCESS
from django.conf import settings
from django.urls import path
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views


urlpatterns = [
    
    path('', views.HomeView.as_view(), name='index'),
    path('services/', views.ServiceListView.as_view(), name="services"),
    path('services/<int:pk>/', views.ServiceDetailView.as_view(),
         name="service_details"),
    path('doctors/', views.DoctorListView.as_view(), name="doctors"),
    path('doctors/<int:pk>/', views.DoctorDetailView.as_view(),
         name="doctor_details"),
    path('faqs/', views.FaqListView.as_view(), name="faqs"),
    path('gallery/', views.GalleryListView.as_view(), name="gallery"),
    path('contact/', views.ContactView.as_view(), name="contact"),
     path('pred1', views.pred1, name='pred1'),
    path('analyze', views.analyze, name='analyze'),
    path('analyze', views.analyze, name='analyze'),

    path('registration', views.registration, name='registration'),

     # path('predictionform', views.predictionform, name='predictionform'),

     path('skin', views.skin, name='skin'),
     
     path('uploaded/', views.uploaded, name='uploaded'),

     path('ecom/', views.ecom, name='ecom'),
     path('base/',views.base, name='base'),
     
     path('FAQs/', views.FAQs_view, name='FAQs'),
]
     

    


