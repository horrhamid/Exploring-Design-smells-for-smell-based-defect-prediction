from multiprocessing import Pool
from projects import ProjectName

import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, RFECV, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings
from sklearn.model_selection import GridSearchCV
from functools import partial
import itertools

# Filter Warnings
filterwarnings(action='ignore')


def get_models():
    models = {
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
            'LogisticRegression': LogisticRegression(),
            'BernoulliNaiveBayes': BernoulliNB(),
            'K-NearestNeighbor': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(),
            'RandomForest': RandomForestClassifier(),
            'SupportVectorMachine': SVC(),
            'MultilayerPerceptron': MLPClassifier()
        }

    return models


def get_params():
    params = {
        'LinearDiscriminantAnalysis': {},
        'QuadraticDiscriminantAnalysis': {},
        'LogisticRegression': {'C': list(np.logspace(-4, 4, 3))},
        'BernoulliNaiveBayes': {},
        'K-NearestNeighbor': {},
        'DecisionTree': {'criterion': ['gini', 'entropy'], },
        'RandomForest': {'n_estimators': [10, 100]},
        'SupportVectorMachine': {'C': [0.1, 100]},
        'MultilayerPerceptron': {'hidden_layer_sizes': [(17, 8, 17)],
                                 'activation': ['tanh', 'relu']}
    }

    return params


def get_selection_methods():
    selection_methods = {
        'chi2_20p': SelectPercentile(chi2, percentile=20),
        'chi2_50p': SelectPercentile(chi2, percentile=50),
        'mutual_info_classif_20p': SelectPercentile(mutual_info_classif, percentile=20),
        'mutual_info_classif_50p': SelectPercentile(mutual_info_classif, percentile=50),
        'f_classif_20': SelectPercentile(f_classif, percentile=20),
        'f_classif_50': SelectPercentile(f_classif, percentile=50),
        'recursive_elimination': RFECV(RandomForestClassifier(), min_features_to_select=3, step=1, cv=5, scoring='f1')
    }

    return selection_methods


def get_datasets(path):
    # read from csv
    train_df = pd.read_csv(path + '/dataset/training.csv')
    test_df = pd.read_csv(path + '/dataset/testing.csv')

    # drop null data

    for df in [train_df, test_df]:
        df.dropna(subset=list(df.columns).remove('Bugged'), inplace=True)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if x is True else 0)

    # get training data and test data
    X_train = train_df.drop(['Bugged'], axis=1)
    y_train = train_df['Bugged']

    X_test = test_df.drop(['Bugged'], axis=1)
    y_test = test_df['Bugged']

    return X_train, y_train, X_test, y_test, test_df, train_df


def selection(X_train, y_train, selection_methods):
    selected_data = {}
    selected_features = {}
    features = X_train.columns
    for method_name, method in selection_methods.items():
        selected_data[method_name] = method.fit_transform(X_train, y_train)
        features_mask = method.get_support()
        selected_features[method_name] = np.array(features)[features_mask].tolist()
    selected_data['all'] = X_train
    selected_features['all'] = list(features)

    return selected_data, selected_features


def over_sample_data(y_train, selected_data):
    oversampled_datasets = {method: SMOTE().fit_resample(X_train, y_train) for method, X_train in selected_data.items()}

    return oversampled_datasets


def fit(X, y, models, params, cv=5, n_jobs=1, verbose=1, scoring=None, refit=False):
    grid_searches = {}

    for key in models.keys():
        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=cv, n_jobs=n_jobs, verbose=verbose,
                          scoring=scoring, refit=refit, return_train_score=True)
        gs.fit(X, y)
        grid_searches[key] = gs

    return grid_searches


def score_summary(grid_searches, sort_by='mean_score'):
    def extract_rows(key: str):
        def get_cv_results(cv, params):
            key = "split{}_test_score".format(cv)
            return grid_search.cv_results_[key]

        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            }
            return pd.Series({**params, **d})

        grid_search = grid_searches[key]
        params = grid_search.cv_results_['params']
        get_cv_results_with_params = partial(get_cv_results, params=params)
        scores = np.hstack(list(map(get_cv_results_with_params, range(grid_search.cv))))
        summary = list(map(lambda values:
                           row(key, values[1], values[0]),
                           list(zip(params, scores))))
        return summary

    rows = list(itertools.chain.from_iterable(map(extract_rows, grid_searches.keys())))
    df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
    columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
    columns = columns + [c for c in df.columns if c not in columns]
    return df[columns]


def get_summary(X, y, models, params):
    grid_searches = fit(X, y, models, params)
    return score_summary(grid_searches)


def get_selected_testing(test_df, selected_features):
    features = test_df.columns
    test_y = test_df['Bugged']
    selected_testing_datasets = {
    method: (test_df[test_df.columns.intersection(features)].values, test_y)
    for method, features in selected_features.items()
    }
    return selected_testing_datasets


def calculate_scores(configurations, oversampled_training, selected_testing, models):
    def calculate_score(method_name, training, testing, configuration):
        estimator = models[configuration['estimator']]
        params = {key: val for key, val in configuration.items() if not (val is None or key == 'estimator')}
        estimator.set_params(**params)
        training_X, training_y = training
        estimator.fit(training_X, training_y)
        testing_X, testing_y = testing
        prediction_y = estimator.predict(testing_X)
        scores_dict = {
            'estimator': configuration['estimator'],
            'configuration': str(params),
            'feature_selection': method_name,
            'precision': precision_score(testing_y, prediction_y),
            'recall': recall_score(testing_y, prediction_y),
            'f1-measure': f1_score(testing_y, prediction_y),
            'auc-roc': roc_auc_score(testing_y, prediction_y),
            'brier score': brier_score_loss(testing_y, prediction_y)
        }
        return scores_dict

    method_names = configurations.keys()
    scores_dicts = list(map(lambda method_name:
                            list(map(lambda configuration:
                                     calculate_score(method_name,
                                                     oversampled_training[method_name],
                                                     selected_testing[method_name],
                                                     configuration),
                                     configurations[method_name])), method_names))
    scores_df = [pd.DataFrame(score) for score in scores_dicts]
    scores = pd.concat(scores_df)
    return scores


def get_scores_info():
    return ['min_score',
            'max_score',
            'mean_score',
            'std_score']


def save_data(name, df):
    # read from csv
    path = name+'.csv'
    df.to_csv(path, index=False)


def run(project):
    # Calculate PATH
    base_path = '../datasets/RQ1/configuration_1/'
    category = 'designite/'
    path = base_path + category + project.value[0]

    # Get data
    X_train, y_train, X_test, y_test, test_df, train_df = get_datasets(path)

    params = get_params()
    models = get_models()
    selection_methods = get_selection_methods()

    selected_data, selected_features = selection(X_train, y_train, selection_methods)

    oversampled_datasets = over_sample_data(y_train, selected_data)

    selected_testing = get_selected_testing(test_df, selected_features)

    summaries = {method: get_summary(data[0], data[1], models, params)
                 for method, data in oversampled_datasets.items()}

    top_summaries = {method: summary[:10] for method, summary in summaries.items()}

    configurations = {method: list(map(lambda x: x[1].to_dict(),
                                       top_summary.drop(get_scores_info(),
                                                        axis=1)
                                       .where(pd.notnull(top_summary), None).iterrows()))
                      for method, top_summary in top_summaries.items()}

    df = calculate_scores(configurations, oversampled_datasets, selected_testing, models)
    print(df)
    # Save data
    save_data(project.value[0], df)


if __name__ == "__main__":
    projects = list(ProjectName)
    print(projects[0])
    run(projects[0])
    # with Pool() as p:
    #     p.map(run, projects)
