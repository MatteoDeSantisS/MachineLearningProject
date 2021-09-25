import pandas as pd
import numpy as np
from numpy import std
from statistics import mean
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def pipeline():
    koi_train = pd.read_csv('processed-data/koi_train.csv')
    koi_topredict = pd.read_csv('processed-data/koi_test.csv')


    X_train, y_train = koi_train.iloc[:, 0:24].to_numpy() , koi_train['koi_disposition']

    #best_parameters_found {'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 400}
    parameters = {
        'n_estimators': [300, 400, 500],
        'max_features': [None,'auto', 'sqrt', 'log2'],
        'max_depth': [7,8,9],
        'min_samples_leaf': [5, 10, 20]
    }
    scoring = ['accuracy', 'precision']

    grid_search = GridSearchCV(param_grid = parameters,
                            cv = StratifiedKFold(10), 
                            estimator = RandomForestClassifier(criterion='gini'),
                            verbose = 1,
                            scoring = scoring,
                            refit = 'accuracy')

    grid_search.fit(X_train, y_train)
    print("___________________________________________________________________________________________\n")
    print(grid_search.best_params_, "\n\nAccuracy score with estimated hyperparameters:", grid_search.best_score_)
    print("___________________________________________________________________________________________")

    rfmodel = RandomForestClassifier(n_estimators = grid_search.best_params_['n_estimators'],
                                    max_features = grid_search.best_params_['max_features'],
                                    max_depth = grid_search.best_params_['max_depth'],
                                    min_samples_leaf = grid_search.best_params_['min_samples_leaf'],
                                    random_state = 0)
    rfmodel.fit(X_train, y_train)

    koi_predicted = grid_search.best_estimator_.predict(koi_topredict)
    koi_predicted = pd.DataFrame(koi_predicted, columns=['prediction'])
    koi_result = koi_topredict.join(koi_predicted)
    n_exoplanets = (koi_result['prediction'] == 0).sum()

    print("\nNumber of exoplanets predicted:", n_exoplanets)


if __name__ == "__main__":
    pipeline()
