import django
import json
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from websiteApi import Ml_model
import pandas as pd
import os
import time


@api_view(['GET'])
def getdata(request):
    df = pd.read_csv('data/website.csv')
    cols = list(df.columns[:-1])
    data = {
        "cols": cols
    }
    return Response(data)


@api_view(['POST'])
def prediction(request):
    try:
        data = json.load(request)
        #print(data['data'])
        result = Ml_model.userprediction(data['data'])
        if(result == "incorrect"):
            raise ValueError
        return Response(json.dumps({"result": result}), content_type="application/json")

    except ValueError as e:
        return Response(json.dumps({"result": 'query datatype/dimensions does not match, follow the sample data format'}), content_type="application/json")


@csrf_exempt
@api_view(["POST"])
def uploadFile(request):

    try:
        data = json.load(request)
        df = pd.DataFrame(data=data["data"])
        df.dropna(inplace=True)
        df_copy = df.copy()
        ans = Ml_model.predictFromCSV(df_copy)
        print(ans)
        if ans == 'incorrect':
            print("innninisdf")
            raise ValueError
        df["result"] = ans
        #print(df['result'])
        encode = {0:"Benigna", 1:"Maligna"}
        df["result"] = df["result"].map(encode)
        path = os.getcwd() + r'\data\results.csv'
        df.to_csv(path, index=False, header=True)
        return Response(json.dumps({"result": "ok", "created": 0, "open": 0}), content_type="application/json")

    except ValueError as e:
        print("error returning")
        return Response(json.dumps({"result":"query datatype/dimensions does not match, follow the sample data format..", "created" : 1, "open" : 1}), content_type="application/json")


@csrf_exempt
def downloadsample(request):
    path = os.getcwd()
    data = open(os.path.join(path, 'data/sample.csv'),'r').read()
    resp = django.http.HttpResponse(data, content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename = sample.csv'
    return resp

@csrf_exempt
def downloadResult(request):
    path = os.getcwd() + r'\data\results.csv'
    data = open(os.path.join(path), 'r').read()
    resp = django.http.HttpResponse(data, content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename = "{}"'.format(str(time.ctime())+"_Result.csv")
    return resp