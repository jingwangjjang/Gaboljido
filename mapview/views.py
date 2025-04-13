from django.shortcuts import render


def show_map(request):
    return render(request, 'mapview/map.html')

