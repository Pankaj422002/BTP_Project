from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse

# Create your views here.
def HomePageHandler(request, *kwarg, **kwargs):
    return render(request, template_name='homepage/home.html')
