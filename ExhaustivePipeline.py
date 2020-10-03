import itertools


class ExhaustivePipeline:
    def __init__(
        self, df, n_k, n_threads,
        feature_pre_selector, feature_pre_selector_kwargs,
        feature_selector, feature_selector_kwargs,
        preprocessor, preprocessor_kwargs,
        classifier, classifier_kwargs
    ):
        '''
        df: pandas dataframe. Rows represent samples, columns represent features (e.g. genes).
        df should also contain three columns:
            -  "Class": binary values associated with target variable;
            -  "Dataset": id of independent dataset;
            -  "Dataset type": "Training", "Filtration" or "Validation".

        n_k: pandas dataframe. Two columns must be specified:
            -  "n": number of features for feature selection
            -  "k": tuple size for exhaustive search
        '''

        self.df = df
        self.n_k = n_k
        self.n_threads = n_threads

        self.feature_pre_selector = feature_pre_selector
        self.feature_pre_selector_kwargs = feature_pre_selector_kwargs

        self.feature_selector = feature_selector
        self.feature_selector_kwargs = feature_selector_kwargs

        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs

    def run(self):
        # First, pre-select features
        features = self.feature_pre_selector(self.df, **self.feature_pre_selector_kwargs)
        df_pre_selected = self.df[features + ["Class", "Dataset", "Dataset type"]]

        # Start iterating over n, k pairs
        for n, k in zip(self.n_k["n"], self.n_k["k"]):
            features = self.feature_selector(df_pre_selected, n, **self.feature_selector_kwargs)
            df_selected = df_pre_selected[features + ["Class", "Dataset", "Dataset type"]]

            # TODO: this loop should be run in multiple processes
            for feature_subset in itertools.combinations(features, k):
                df_train = df_selected.loc[df_selected["Dataset type"] == "Training", feature_subset + ["Class"]]
                X_train = df_train.drop(columns=["Class"]).to_numpy()
                y_train = df_train["Class"].to_numpy()

                self.preprocessor.fit(X_train, **self.preprocessor_kwargs)
                X_train = self.preprocessor.transform(X_train)

                # TODO: train classifier (with cross-validated parameters)
                # TODO: evaluate performance on EACH dataset and store results


def feature_pre_selector_template(df, **kwargs):
    '''
    Input expression dataframe, return list of features
    TODO: special function which load genes from specified file
    '''
    pass


def feature_selector_template(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    pass


class Preprocessor_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  transform
    Any sklearn classifier preprocessor be suitable
    '''
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass


class Classifier_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  predict
    Any sklearn classifier will be suitable
    '''
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
