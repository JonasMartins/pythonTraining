from django.shortcuts import render
from django.views.generic import ListView

from core.models import Movie
# Create your views here.

class MovieList(ListView):
  model = Movie
