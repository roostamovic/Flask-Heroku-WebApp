from flask import Flask, render_template, request, redirect, url_for

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

@app.route("/")
def index():

    return render_template("index.html")                

@app.route("/predict",methods = ["POST"])
def predictApp():

    df = pd.read_csv('final_data.csv', index_col=0)
    X = df.iloc[:, :7]  # we only take the first two features.
    y = df[['status']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=4)

    pipeline_knn = Pipeline([('scaler2', StandardScaler()),
                            ('pca2', PCA(n_components=2)),
                            ('knn', KNeighborsClassifier(n_neighbors=11))])

    pipeline_svm = Pipeline([('scaler1', StandardScaler()),
                            ('pca1', PCA(n_components=2)),
                            ('svm', SVC())])

    pipeline_knn.fit(X_train, y_train)
    pred_knn = pipeline_knn.predict(X_test)
    score_knn = round(accuracy_score(y_test, pred_knn), 5)

    pipeline_svm.fit(X_train, y_train)
    pred_svm = pipeline_svm.predict(X_test)
    score_svm = round(accuracy_score(y_test, pred_svm), 5)

    init_features = [float(x) for x in request.form.values()]

    if init_features[-2] != 0:
        init_features = np.append(init_features, init_features[0] / init_features[-2])
    else:
        init_features = np.append(init_features, init_features[0])

    operating = np.array([1])
    acquired = np.array([2])
    closed = np.array([3])
    ipo = np.array([4])
    
    final_features = [np.array(init_features)]

    predict_knn = pipeline_knn.predict(final_features)
    predict_svm = pipeline_svm.predict(final_features)

    return render_template("index.html", operating=operating, acquired=acquired, closed=closed, ipo=ipo, accuracy_knn=score_knn*100, predict_knn=predict_knn, accuracy_svm=score_svm*100, predict_svm=predict_svm)                
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug = True)