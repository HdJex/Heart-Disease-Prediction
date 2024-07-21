from django.contrib import admin
from .models import (Slider, Service, Item, Doctor, Expertize, Faq, Gallery,Registration)
from . models import Product,Offer

admin.site.register(Slider)
admin.site.register(Service)
admin.site.register(Item)
admin.site.register(Doctor)
admin.site.register(Expertize)
admin.site.register(Faq)
admin.site.register(Gallery)
admin.site.register(Registration)

class RegisterAdmin(admin.ModelAdmin):
    list_display=('name','email','password','cpassword')

class ProductAdmin(admin.ModelAdmin):
    list_display = ('name','price','stock')


class OfferAdmin(admin.ModelAdmin):
    list_display = ('code','discount')


admin.site.register(Product,ProductAdmin)
admin.site.register(Offer ,OfferAdmin)