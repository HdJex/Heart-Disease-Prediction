from django.db import models
from hospital.models import Doctor
from django.utils import timezone
from django.contrib.auth.models import User

class Appointment(models.Model):
    time_choices = (
        ('morning', "Morning"),
        ('afternoon', "Afternoon"),
        ('evening', "Evening")
    )
    name = models.CharField(max_length=120)
    phone = models.CharField(max_length=20)
    email = models.EmailField()
    doctor = models.ForeignKey(
        Doctor, on_delete=models.CASCADE, related_name='appointments')
    date = models.DateField(default=timezone.now)
    time = models.CharField(choices=time_choices, max_length=10)
    note = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.name}-{self.doctor.name}"

STATE_CHOICES = (
    ('Andhra Pradesh','Andhra Pradesh'), 	
 	('Arunachal Pradesh','Arunachal Pradesh'), 	
 	('Assam','Assam'), 	
 	('Bihar','Bihar'),	
 	('Chhattisgarh','Chhattisgarh'), 	
 	('Goa','Goa'),
 	('Gujarat','Gujarat'), 	
 	('Haryana','Haryana'), 	
 	('Himachal Pradesh','Himachal Pradesh'), 
 	('Jammu Kashmir','Jammu Kashmir'), 	
 	('Jharkhand','Jharkhand'), 	
 	('Karnataka', 'Karnataka'), 
 	('Kerala','Kerala'), 	
 	('Madhya Pradesh','Madhya Pradesh'), 	
 	('Maharashtra','Maharashtra'), 	
	('Manipur','Manipur'), 	
	('Meghalaya','Meghalaya'), 	
 	('Mizoram','Mizoram'), 	
 	('Nagaland','Nagaland'), 	
 	('Odisha','Odisha'), 	
 	('Punjab','Punjab'), 	
 	('Rajasthan','Rajasthan'), 	
 	('Sikkim','Sikkim'), 	
 	('Tamil Nadu','Tamil Nadu'), 
 	('Telangana','Telangana'), 	
 	('Tripura','Tripura'), 	
 	('Uttar Pradesh','Uttar Pradesh'), 	
 	('Uttarakhand','Uttarakhand'), 	
 	('West Bengal','West Bengal')
)
class Customer(models.Model):
 user = models.ForeignKey(User, on_delete=models.CASCADE)
 name = models.CharField(max_length=200)
 state = models.CharField(choices=STATE_CHOICES, max_length=150)
 city = models.CharField( max_length=50)
 locality = models.CharField(max_length=200)
 zipcode = models.IntegerField()

 def _str_(self):
  return str(self.id)