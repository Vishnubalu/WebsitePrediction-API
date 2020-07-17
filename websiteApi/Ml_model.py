import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import pickle

def create_model(features, target):
    scaler = StandardScaler()
    sclr_train = features
    scaler.fit(sclr_train)
    scaled_features = scaler.transform(sclr_train)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.30, random_state=100)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    with open ("data/model.pkl", "wb") as file:
        pickle.dump(knn,file)


def columnEncoding():
    df = pd.read_csv('data/website.csv')
    target = {"Benigna": 0, "Maligna": 1}
    df['TIPO'] = df['TIPO'].map(target)
    y = df['TIPO']
    x = df.drop(columns=['TIPO'], axis=0)
    beforecols = list(x.columns)
    mod_x = pd.get_dummies(x, columns=['URL', 'DOMAIN_NAME', 'CHARSET', 'SERVER', 'CACHE_CONTROL', 'WHOIS_COUNTRY',
                                       'WHOIS_STATE_CITY'])
    aftercols = list(mod_x.columns)
    return (beforecols, aftercols, mod_x, y)

def predition(mod_x, y, test):
    try:
        with open ('data/model.pkl', 'rb') as file:
            model = pickle.load(file)
    except:
        print("in except")
        create_model(mod_x,y)
    with open('data/model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("model loaded")
    pred = model.predict(test)
    return pred



def userprediction(request):
    (beforecols, aftercols, mod_x, y) = columnEncoding()
    given_data = pd.DataFrame([[0] * (len(beforecols))], columns=beforecols)
    data = request
    for key in data:
        given_data[key] = data[key]
    mod_given = pd.get_dummies(given_data,
                               columns=['URL', 'DOMAIN_NAME', 'CHARSET', 'SERVER', 'CACHE_CONTROL', 'WHOIS_COUNTRY',
                                        'WHOIS_STATE_CITY'])
    test = pd.DataFrame([[0] * (len(aftercols))], columns=aftercols)
    for col in list(mod_given.columns):
        if col in aftercols:
            test[col] = mod_given[col]
        else:
            return 'incorrect'
    pred = predition(mod_x, y, test)
    if pred[0] == 0:
        return "Benigna"
    else:
        return "Maligna"

def predictFromCSV(data):

    (beforecols, aftercols, mod_x, y) = columnEncoding()
    data.columns = map(str.upper, data.columns)
    print(data.columns)
    try:
        mod_given = pd.get_dummies(data, columns=['URL', 'DOMAIN_NAME', 'CHARSET', 'SERVER', 'CACHE_CONTROL', 'WHOIS_COUNTRY',
                             'WHOIS_STATE_CITY'])
    except:
        print('except')
        return 'incorrect'

    for col in aftercols:
        if col not in list(mod_given.columns):
            return 'incorrect'
    # print(aftercols)

    data = mod_given[aftercols]
    print("after")
    scaler = StandardScaler()
    sclr_train = data
    scaler.fit(sclr_train)
    scaled_features = scaler.transform(sclr_train)
    pred = predition(mod_x, y, scaled_features)
    return pred












