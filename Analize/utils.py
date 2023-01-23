import itertools
import logging
import os
from functools import partial

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GridSearchCV

from config import Config
from projects import ProjectName


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: {0}".format(missing_params))
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.general_log = logging.getLogger(__name__)

    def fit(self, X, y, cv=10, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            self.general_log.info("Running GridSearchCV for {0}".format(key))
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose,
                              scoring=scoring, refit=refit, return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):

        def extract_rows(key: str):
            def get_cv_results(cv, params):
                key = "split{}_test_score".format(cv)
                return grid_search.cv_results_[key].reshape(len(params), 1)

            def row(key, scores, params):
                d = {
                    'estimator': key,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                }
                return pd.Series({**params, **d})

            grid_search = self.grid_searches[key]
            params = grid_search.cv_results_['params']
            get_cv_results_with_params = partial(get_cv_results, params=params)
            scores = np.hstack(list(map(get_cv_results_with_params, range(grid_search.cv))))
            summary = list(map(lambda values:
                               row(key, values[1], values[0]),
                               list(zip(params, scores))))
            return summary

        rows = list(itertools.chain.from_iterable(map(extract_rows, self.grid_searches.keys())))
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        return df[columns]

    @staticmethod
    def get_scores_info():
        return ['min_score',
                'max_score',
                'mean_score',
                'std_score']


class FeatureSelectionHelper:
    def __init__(self, methods, features):
        self.methods = methods
        self.selected_features = {}
        self.selected_data = {}
        self.features = features

    def select(self, X, y):
        for method_name, method in self.methods.items():
            self.selected_data[method_name] = method.fit_transform(X, y)
            features_mask = method.get_support()
            self.selected_features[method_name] = np.array(self.features)[features_mask].tolist()
        self.selected_data['all'] = X
        self.selected_features['all'] = list(self.features)

    def get_selected_features(self):
        return self.selected_features

    def get_selected_dataset(self):
        return self.selected_data


def pandas_to_latex(df_table, latex_file, vertical_bars=False, right_align_first_column=True, header=True, index=False,
                    escape=False, multicolumn=False, **kwargs) -> None:
    """
    Function that augments pandas DataFrame.to_latex() capability.
    :param df_table: dataframe
    :param latex_file: filename to write latex table code to
    :param vertical_bars: Add vertical bars to the table (note that latex's booktabs table format that pandas uses is
                          incompatible with vertical bars, so the top/mid/bottom rules are changed to hlines.
    :param right_align_first_column: Allows option to turn off right-aligned first column
    :param header: Whether or not to display the header
    :param index: Whether or not to display the index labels
    :param escape: Whether or not to escape latex commands. Set to false to pass deliberate latex commands yourself
    :param multicolumn: Enable better handling for multi-index column headers - adds midrules
    :param kwargs: additional arguments to pass through to DataFrame.to_latex()
    :return: None
    """
    n = len(df_table.columns) + int(index)

    if right_align_first_column:
        cols = 'r' + 'c' * (n - 1)
    else:
        cols = 'c' * n

    if vertical_bars:
        # Add the vertical lines
        cols = '|' + '|'.join(cols) + '|'

    latex = df_table.to_latex(escape=escape, index=index, column_format=cols, header=header, multicolumn=multicolumn,
                              **kwargs)

    if vertical_bars:
        # Remove the booktabs rules since they are incompatible with vertical lines
        latex = re.sub(r'\\(top|mid|bottom)rule', r'\\hline', latex)

    # Multicolumn improvements - center level 1 headers and add midrules
    if multicolumn:
        latex = latex.replace(r'{l}', r'{c}')

        offset = int(index)
        midrule_str = ''
        for i, col in enumerate(df_table.columns.levels[0]):
            indices = np.nonzero(np.array(df_table.columns.codes[0]) == i)[0]
            hstart = 1 + offset + indices[0]
            hend = 1 + offset + indices[-1]
            midrule_str += rf'\cmidrule(lr){{{hstart}-{hend}}} '

        # Ensure that headers don't get colored by row highlighting
        midrule_str += r'\rowcolor{white}'

        latex_lines = latex.splitlines()
        latex_lines.insert(3, midrule_str)
        latex = '\n'.join(latex_lines)

    with open(latex_file, 'w') as f:
        f.write(latex)


class FindBestClassifier():
    def __init__(self, base, dataset):
        self.path = Config.get_work_dir_path(os.path.join("paper", base, dataset))
        assert os.path.exists(self.path)
        self.scores = ["precision", "recall", "f1-measure", "auc-roc", "brier score"]
        self.best_df = self._find()

    def _find(self):
        projects = list(map(lambda x: x.github(), list(ProjectName)))
        best = []
        for project in projects:
            try:
                scores_path = os.path.join(self.path, project, "scores.csv")
                df = pd.read_csv(scores_path)
                cond = df['feature_selection'] == "all"
                df = df.loc[cond]
                best += list(map(lambda score: (
                    score,
                    df["estimator"].iloc[df[score].argmax()],
                    df["configuration"].iloc[df[score].argmax()]
                ), self.scores))
            except:
                print("Didn't get project {}".format(project))

        return pd.DataFrame(best, columns=["Score", "Estimator", "Configuration"])

    def get_best_estimator(self, score="f1-measure", include_configuration=False):
        if score not in self.scores:
            raise ValueError("Score not found")
        cond = self.best_df['Score'] == score
        df = self.best_df.loc[cond]
        index = ["Estimator"]
        if include_configuration:
            index.append("Configuration")
        pivot_df = df.pivot_table(index=index, aggfunc='size').reset_index()
        pivot_df.rename(columns={pivot_df.columns[-1]: "Sum"}, inplace=True)
        results = []

        def get_max(res_list):
            result = pivot_df.iloc[[pivot_df['Sum'].idxmax()]]
            s = pivot_df.max()['Sum']
            result.drop(columns=["Sum"], axis=1, inplace=True)
            res_list += [tuple(x) for x in result.to_numpy()]
            pivot_df.drop(pivot_df['Sum'].idxmax(), axis=0, inplace=True)
            pivot_df.reset_index(drop=True, inplace=True)
            return s, res_list

        base = -1
        while True:
            m, results = get_max(results)
            if base == -1:
                base = m
            if m < base:
                return results[:-1]

if __name__ == '__main__':
    res = FindBestClassifier("analysis", "designite_fowler").get_best_estimator(include_configuration=True)
