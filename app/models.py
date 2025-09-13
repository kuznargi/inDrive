from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    face_encoding = models.BinaryField(null=True, blank=True) 
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[("male", "Male"), ("female", "Female")], null=True, blank=True)
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.username

