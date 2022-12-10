import pickle
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import tsfresh
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from SODA import SelfOrganisedDirectionAwareDataPartitioning
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute


def data_filter(data, fc=1e3):

    fs = 24e3  # Frequencia de amostragem

    L, W = data.shape
    ts_len = int(data[:, 1].max())
    data_ts_num = L // ts_len

    time_id = [i for i in range(1, ts_len + 1)]
    new_data = np.zeros((ts_len * data_ts_num, W))

    for ts_id in range(data_ts_num):
        target = data[ts_id * ts_len, -1]
        index = data[ts_id * ts_len, 0]

        new_data[ts_id * ts_len : (ts_id + 1) * ts_len, 0] = ts_len * [index]
        new_data[ts_id * ts_len : (ts_id + 1) * ts_len, 1] = time_id
        new_data[ts_id * ts_len : (ts_id + 1) * ts_len, -1] = ts_len * [target]

        for sensor in range(3):
            X = data[ts_id * ts_len : (ts_id + 1) * ts_len, sensor + 2].copy()

            sos = signal.butter(2, 2 * np.pi * fc, "low", fs=fs, output="sos")
            X_filtered = signal.sosfilt(sos, X)

            new_data[ts_id * ts_len : (ts_id + 1) * ts_len, sensor + 2] = X_filtered

    ### PLOTAR DIAGRAMA DE BODE
    b, a = signal.butter(2, 2 * np.pi * fc, "low", analog=True)
    w, h = signal.freqs(b, a)

    fig = plt.figure(figsize=(16, 6))
    ax = fig.subplots(1, 2)

    ax[0].semilogx(w / (2 * np.pi), 20 * np.log10(abs(h)))
    ax[1].semilogx(w / (2 * np.pi), np.angle(h) * 180 / np.pi)

    ax[0].set_xlabel("Frequência [Hz]", fontsize=15)
    ax[0].set_ylabel("Amplitude [dB]", fontsize=15)
    ax[1].set_xlabel("Frequência [Hz]", fontsize=15)
    ax[1].set_ylabel("Fase [°]", fontsize=15)

    ax[0].margins(0, 0.1)
    ax[1].margins(0, 0.1)
    ax[0].grid(which="both", axis="both")
    ax[1].grid(which="both", axis="both")

    plt.savefig("Figures/Bode.png", bbox_inches="tight")

    return new_data


class LathesModel(object):
    """Lathes Cutting Tool Model Class

    Parameters
    ----------
    N_PCs: int, default=3
        Number of components to keep in PCA.
    clf: classifier, default=MLPClassifier
        sklearn binary classifier
    n_jobs: int, default=4
        The number of processes to use for parallelization in tsfresh
    granularity: float, default=3
        SODA granularity, sensibility factor for data partitioning module
    percent: float, default=50
        purity percent for grouping algorithm, must be within (50, 100) interval
        percent=50 means hard voting
    selection_type: string, default="union"
        TSFRESH feature selection method, can be [union, intersection]

    Attributes
    ----------
    N_PCs_: int
        Number of components to keep in PCA.
    n_jobs_: int
        The number of processes to use for parallelization in tsfresh
    granularity_: float
        SODA granularity, sensibility factor for data partitioning module
    percent_: float
        purity percent for grouping algorithm
    nan_columns_: list
        name of columns with NaN values
    valid_columns_: list
        name of columns without NaN values
    target_: np.array
        target for each time serie used to fit the model
    relevance_table_: pd.DataFrame
        relevance table after KS test
    relevant_features_: pd.Series
        name of features selected by TSFRESH hypothesis test
    selected_columns_: pd.Index
        name of features selected by TSFRESH hypothesis test
    kind_to_fc_parameters_: dict
        dictionary with features selected by TSFRESH
        keys = sensor names
    X_selected_: pd.DataFrame
        train data set features after TSFRESH selection
    X_projected_: np.array
        train data set projected in Principal Components
    variation_kept_: np.array
        Percentage of variance explained by each of the selected components.
    SODA_output_: dict
        Dictionary with SODA output
    SODA_IDX_: np.array
        array with labels given by SODA algorithm
    classifiers_label_: np.array
        array with labels given by Grouping Algorithm
    GA_results_: dict
        'Data_Clouds': int
            number of Data Clouds
        'Good_Tools_Groups': int
            number of Adequate Condition Data Clouds
        'Worn_Tools_Groups': int
            number of Inadequate Condition Data Clouds
        'Samples': int
            number of Samples
    X_test_selected_: np.array
        test data set features selected by tsfresh
    X_test_projected_: np.array
        test data set projected in Principal Components
    n_timeseries_ : int
        number of timeseries in train dataset
    n_measures_: int
        number of measurements in timeseries
    n_sensors_: int
        number of sensors in dataset
    already_fitted_: bool
        if model has already been fitted
            already_fitted = True
        else
            already_fitted = False
    already_tested_: bool
        if model has already been tested
            already_tested = True
        else
            already_test = False
    fit_time_: datetime
        time to fit model
    tsfresh_time_: datetime
        time to fit tsfresh
    predict_time_: datetime
        time of last prediction
    tsfresh_predict_time_: datetime
        time of last prediction tsfresh
    one_class_: bool
        if one_class_ == True:
            model has only one class and has to be fitted with higher granularity
        else
            model has more than one class

    clf: sklearn classifier
        sklearn binary classifier
    scaler: sklearn.preprocessing.MinMaxScaler
        scaler to normalize input data
    pca_scaler: sklearn.preprocessing.StandardScaler
        scaler to standardize selected features
    pca: sklearn.decomposition.PCA
        pca fitted model
    """

    def __init__(self, N_PCs=3, clf="None", n_jobs=0, granularity=3, percent=50, selection_type="union"):

        self.N_PCs_ = N_PCs
        self.granularity_ = granularity
        self.n_jobs_ = n_jobs
        self.percent_ = percent
        self.selection_type_ = selection_type
        if clf == "None":
            self.clf = MLPClassifier(alpha=1, max_iter=500)
        else:
            self.clf = clf

        self.already_fitted_ = False
        self.already_tested_ = False
        self.one_class_ = False

    def reset(self):
        """Reset model
        This function will reset the model, if methods 'fit_after_tsfresh' or 'predict_after_tsfresh'
        are called it will execute the whole fit or prediction methods again"""
        self.already_fitted_ = False
        self.already_tested_ = False
        self.one_class_ = False

    ### Fitting Methods

    def _normalization(self, X, y):
        """Normalize input data in fit stage"""
        L, W = X.shape

        self.n_measures_ = int(X[:, 1].max())
        self.n_timeseries_ = int(L / self.n_measures_)
        self.n_sensors_ = int(W - 2)
        self.target_ = y[:: self.n_measures_]

        info = X[:, 0:2]
        data = X[:, 2:]

        # self.scaler = MinMaxScaler()
        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        df = pd.DataFrame(np.concatenate((info, data), axis=1), columns=["id", "time"] + ["Sensor_" + str(x) for x in range(1, self.n_sensors_ + 1)])
        return df

    def _tsfresh_extraction(self, X):

        param = EfficientFCParameters()
        default_fc_parameters = {
            "fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(self.n_measures // 2))]
        }
        param["fft_coefficient"] = default_fc_parameters["fft_coefficient"]

        """Feature Extraction in fit stage
        After extraction columns with NaN values are dropped"""
        extracted_features = tsfresh.extract_features(X, column_id="id", column_sort="time", n_jobs=self.n_jobs_, default_fc_parameters=param,)

        features = extracted_features.columns
        self.nan_columns_ = []
        self.valid_columns_ = []
        for col in features:
            if extracted_features.loc[:, col].hasnans:
                self.nan_columns_.append(col)
            else:
                self.valid_columns_.append(col)

        return extracted_features.drop(self.nan_columns_, axis=1)

    def _tsfresh_selection_3class(self, X):
        """Feature Selection for fit stage"""
        # First Selection
        y = self.target_.copy()
        y[y == 2] = 1

        y = pd.Series(y, index=X.index)

        self.relevance_table_ = calculate_relevance_table(X, y)

        self.relevant_features_ = self.relevance_table_[self.relevance_table_.relevant].feature

        X_selected_ = X.loc[:, self.relevant_features_]

        selected_columns_ = X_selected_.columns

        # Second Selection
        idx = np.squeeze(np.argwhere(self.target_ != 0))

        X2 = X.iloc[idx]
        y = pd.Series(self.target_[idx], index=X2.index)

        self.relevance_table_2_ = calculate_relevance_table(X2, y)

        self.relevant_features_2_ = self.relevance_table_2_[self.relevance_table_2_.relevant].feature

        X_selected_2_ = X2.loc[:, self.relevant_features_2_]

        selected_columns_2_ = X_selected_2_.columns

        if self.selection_type_ == "union":
            union = set(selected_columns_) | set(selected_columns_2_)
            self.selected_columns_ = list(union)
            self.kind_to_fc_parameters_ = tsfresh.feature_extraction.settings.from_columns(self.selected_columns_)
        elif self.selection_type_ == "intersection":
            inter = set(selected_columns_) & set(selected_columns_2_)
            self.selected_columns_ = list(inter)
            self.kind_to_fc_parameters_ = tsfresh.feature_extraction.settings.from_columns(self.selected_columns_)

        self.selected_columns_.sort()

        self.X_selected_ = X.loc[:, self.selected_columns_]

    def _pca(self):
        """PCA calculation and projection for fit stage"""
        self.pca_scaler = StandardScaler()
        X_scaled = self.pca_scaler.fit_transform(self.X_selected_)

        self.pca = PCA(n_components=self.N_PCs_)
        self.pca.fit(X_scaled)

        self.X_projected_ = self.pca.transform(X_scaled)

        self.variation_kept_ = self.pca.explained_variance_ratio_ * 100

    def _soda(self):
        """SODA Data Partitioning Algorithm for fit stage"""
        Input = {"GridSize": self.granularity_, "StaticData": self.X_projected_, "DistanceType": "euclidean"}
        self.SODA_output_ = SelfOrganisedDirectionAwareDataPartitioning(Input)

        self.SODA_IDX_ = self.SODA_output_["IDX"]

    def _grouping_algorithm(self):
        """Grouping Algorithm for fit stage"""
        #### Program Matrix's and Variables ####
        n_DA_planes = np.max(self.SODA_IDX_)
        Percent = np.zeros((int(n_DA_planes), 3))
        n_IDs_per_gp = np.zeros((int(n_DA_planes), 3))
        n_tot_Id_per_DA = np.zeros((int(n_DA_planes), 1))
        decision = np.zeros(int(n_DA_planes))
        n_gp0 = 0
        n_gp1 = 0
        n_gp2 = 0

        #### Definition Percentage Calculation #####

        for i in range(self.target_.shape[0]):
            if self.target_[i] == 0:
                n_IDs_per_gp[int(self.SODA_IDX_[i] - 1), 0] += 1
            elif self.target_[i] == 1:
                n_IDs_per_gp[int(self.SODA_IDX_[i] - 1), 1] += 1
            elif self.target_[i] == 2:
                n_IDs_per_gp[int(self.SODA_IDX_[i] - 1), 2] += 1

            n_tot_Id_per_DA[int(self.SODA_IDX_[i] - 1)] += 1

        for i in range(int(n_DA_planes)):

            Percent[i, 0] = (n_IDs_per_gp[i, 0] / n_tot_Id_per_DA[i]) * 100
            Percent[i, 1] = (n_IDs_per_gp[i, 1] / n_tot_Id_per_DA[i]) * 100
            Percent[i, 2] = (n_IDs_per_gp[i, 2] / n_tot_Id_per_DA[i]) * 100

        #### Using Definition Percentage as Decision Parameter ####

        for i in range(Percent.shape[0]):

            if Percent[i, 0] > self.percent_:
                n_gp0 += 1
                decision[i] = 0
            elif Percent[i, 1] > self.percent_:
                n_gp1 += 1
                decision[i] = 1
            elif Percent[i, 2] > self.percent_:
                n_gp2 += 1
                decision[i] = 2

                #### Defining labels

        self.classifiers_label_ = []

        for i in range(len(self.SODA_IDX_)):
            self.classifiers_label_.append(decision[int(self.SODA_IDX_[i] - 1)])

        ### Printig Analitics results

        self.GA_results_ = {
            "Data_Clouds": n_DA_planes,
            "Good_Tools_Groups": n_gp0,
            "Imminence_Tools_Groups": n_gp1,
            "Worn_Tools_Groups": n_gp2,
            "Samples": int(len(self.SODA_IDX_)),
        }

    ### Prediction Methods

    def _predict_normalization(self, X):
        """Normalize input data for prediction stage
        This step is executed using 'scaler' fitted in .fit"""
        info = X[:, 0:2]
        data = X[:, 2:]

        L, W = X.shape

        data = self.scaler.transform(data)

        df = pd.DataFrame(np.concatenate((info, data), axis=1), columns=["id", "time"] + ["Sensor_" + str(x) for x in range(1, self.n_sensors_ + 1)])

        return df

    def _predict_tsfresh_extraction(self, X):
        """Feature Extraction for prediction stage
        This step is executed using 'kind_to_fc_parameters_' constructed in .fit"""
        columns = []
        for i, x in enumerate(self.kind_to_fc_parameters_):
            aux = pd.DataFrame(np.hstack((X.loc[:, :"time"].values, X.loc[:, x].values.reshape((-1, 1)))), columns=["id", "time", x])

            aux2 = tsfresh.extract_features(
                aux, column_id="id", column_sort="time", default_fc_parameters=self.kind_to_fc_parameters_[x], n_jobs=self.n_jobs_
            )

            for j in range(len(aux2.columns.tolist())):
                columns.append(aux2.columns.tolist()[j])

            if i == 0:
                extracted_features = np.array(aux2.values)
            else:
                extracted_features = np.hstack((extracted_features, aux2.values))

        final_features = pd.DataFrame(extracted_features, columns=columns)
        self.X_test_selected_ = impute(final_features[self.selected_columns_])

    def _predict_pca(self):
        """Project predict data using PCA fitted in .fit"""
        X_scaled = self.pca_scaler.transform(self.X_test_selected_)

        self.X_test_projected_ = self.pca.transform(X_scaled)

    ### Main Methods

    def fit(self, X, y):
        """Fit the model with X and target y

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data, in following format

            | ID            | Time_ID     |  Sensor 1 | ... |  Sensor n |
            |---------------|-------------|-----------| ... |-----------|
            | 1             | 1           |  1.91     | ... | -1.03     |
            | 1             | 2           |  1.06     | ... |  1.17     |
            | ...           | ...         | ...       | ... | ...       |
            | 1             | n_measures_ |  0.48     | ... | -1.69     |
            | 2             | 1           |  0.78     | ... |  1.45     |
            | 2             | 2           | -0.21     | ... |  0.46     |
            | ...           | ...         | ...       | ... | ...       |
            | 2             | n_measures_ | -1.18     | ... | -0.09     |
            | ...           | ...         | ...       | ... | ...       |
            | n_timeseries_ | 1           | -0.85     | ... |  0.07     |
            | n_timeseries_ | 2           | -1.18     | ... |  0.64     |
            | ...           | ...         | ...       | ... | ...       |
            | n_timeseries_ | n_measures_ | -0.83     | ... | -0.97     |

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """

        start = datetime.now()

        X_norm = self._normalization(X, y)

        X_extracted = self._tsfresh_extraction(X_norm)

        self._tsfresh_selection_3class(X_extracted)

        self.tsfresh_time_ = datetime.now() - start

        self._pca()

        self._soda()

        self._grouping_algorithm()

        try:
            # self.clf.fit(self.X_projected_, self.classifiers_label_)
            self.clf.fit(self.X_projected_, self.target_)

            self.one_class_ = False
        except:
            self.one_class_ = True

        self.already_fitted_ = True

        self.fit_time_ = datetime.now() - start

    def fit_predict(self, X, y):
        """Fit the model with X and target y and predict the target after that

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data

        Returns
        -------
        y_pred : np.array shape(n_timeseries,)
            Predictions for training data
        """
        self.fit(X, y)

        y_pred = self.clf.predict(self.X_projected_)

        return y_pred

    def predict(self, X):
        """Predict using the trained model

        Parameters
        ----------
        X : np.array, shape (n_timeseries_*n_measures_, n_sensors+2)
            The input data.

        Returns
        -------
        y_pred : np.array (n_timeseries,)
            The predicted class for each timeseries presented to the model.
        """

        if self.one_class_:
            return
        else:
            start = datetime.now()

            X_norm = self._predict_normalization(X)

            self._predict_tsfresh_extraction(X_norm)

            self.tsfresh_predict_time_ = datetime.now() - start

            self._predict_pca()

            y_pred = self.clf.predict(self.X_test_projected_)

            self.already_tested_ = True

            self.predict_time_ = datetime.now() - start

        return y_pred

    def fit_after_tsfresh(self, X, y):
        """Fit the model with X and target y after TSFRESH extraction
        and selection already had been performed.
        This method is useful for train the model after change some parameter
        as N_PCs, granularity or classifier without the need of execute the TSFRESH
        extraction module again.

        If the model wasn't fitted before the model will be fitted from the start.

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """
        if self.already_fitted_:
            start = datetime.now()
            self._pca()

            self._soda()

            self._grouping_algorithm()

            try:
                # self.clf.fit(self.X_projected_, self.classifiers_label_)
                self.clf.fit(self.X_projected_, self.target_)
                self.one_class_ = False
            except:
                self.one_class_ = True

            self.fit_time_ = datetime.now() - start + self.tsfresh_time_
        else:
            print("Fitting from start!")
            self.fit(X, y)

    def predict_after_tsfresh(self, X):
        """Predict using trained model same dataset that was last predicted.
        This method is useful for predict the model after change some parameter
        as N_PCs, granularity or classifier without the need of execute the TSFRESH
        extraction module again.

        If the model wasn't predicted before the model will predict from the start.

        Parameters
        ----------
        X : array-like, shape (n_timeseries_*n_measures_, n_sensors+2)
            Training data

        y : np.array, shape (n_timeseries_*n_measures_)
            Target for training data
        """
        if self.already_tested_:
            if self.one_class_:
                return
            else:
                start = datetime.now()
                self._predict_pca()

                y_pred = self.clf.predict(self.X_test_projected_)

                self.predict_time_ = datetime.now() - start + self.tsfresh_predict_time_

        else:
            print("Predicting from start!")
            y_pred = self.predict(X)

        return y_pred

    def change_hyperparams(self, params):
        """Change model Hyperparams
        This function is useful to change a model param without the need of reconstruct the model

        Parameters
        ----------
        params: dict
            dictionary with some of the following keys
            'N_PCs': int
                Number of components to keep in PCA.
            'clf': classifier
                sklearn binary classifier
            'n_jobs': int
                The number of processes to use for parallelization in tsfresh
            'granularity': float
                SODA granularity, sensibility factor for data partitioning module
            'percent': float
                purity percent for grouping algorithm, must be within (50, 100) interval
        """

        for p in params:
            if p == "clf":
                exec("self.{} = {}".format(p, params[p]))
            if p == "selection_type":
                exec('self.{}_ = "{}"'.format(p, params[p]))
            else:
                exec("self.{}_ = {}".format(p, params[p]))


input_id = 0
gra = 7
N_PCs = 15

if __name__ == "__main__":

    select_list = ["intersection", "union"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma="scale"),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1, max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    folds = 5

    PATH = "../Input/Input_%i.csv" % (input_id)

    full_data = pd.read_csv(PATH, header=None).values

    L, W = full_data.shape
    full_data = full_data[:, [0, 1, 5, 6, 7, 8]]
    n_measures = int(full_data[:, 1].max())
    n_timeseries = int(L / n_measures)

    unique_id = np.arange(n_timeseries)
    target = full_data[::n_measures, -1]

    kf = StratifiedKFold(n_splits=folds, random_state=12, shuffle=True)

    full_data = data_filter(full_data, 200)

    for it, (train_unique_index, test_unique_index) in enumerate(kf.split(unique_id, target)):
        print("-----> Input -", input_id, "\n-----> Fold -", it, "\n", datetime.now())
        print("Train: {}".format(len(train_unique_index)))
        print("Test: {}".format(len(test_unique_index)))

        train_unique_index.sort()
        test_unique_index.sort()

        L_train = train_unique_index.shape[0]
        train_index = np.zeros(L_train * n_measures, dtype=np.int32)
        for ii in range(L_train):
            train_index[ii * n_measures : (ii + 1) * n_measures] = list(
                range(train_unique_index[ii] * n_measures, (train_unique_index[ii] + 1) * n_measures - (n_measures - n_measures))
            )
        L_test = test_unique_index.shape[0]
        test_index = np.zeros(L_test * n_measures, dtype=np.int32)
        for ii in range(L_test):
            test_index[ii * n_measures : (ii + 1) * n_measures] = list(
                range(test_unique_index[ii] * n_measures, (test_unique_index[ii] + 1) * n_measures - (n_measures - n_measures))
            )

        X_train = full_data[train_index, :-1]
        y_train = full_data[train_index, -1]

        X_test = full_data[test_index, :-1]
        y_test = full_data[test_index, -1]
        y_test = y_test[::n_measures]
        Classifiers_result = {}

        print("-----------------------------")
        print("n_time_series", n_timeseries)
        print("n_measures:", n_measures)
        print("-----------------------------")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)
        print("-----------------------------")

        for st in select_list:
            model = LathesModel(N_PCs=N_PCs, n_jobs=0)
            Classifiers_result = {
                "Nearest Neighbors": {},
                "Linear SVM": {},
                "Decision Tree": {},
                "Random Forest": {},
                "Neural Net": {},
                "AdaBoost": {},
                "Naive Bayes": {},
                "QDA": {},
            }

            for name, clf in zip(names, classifiers):
                Classifiers_result[name] = {"Accuracy": 0, "time": 0}

                params = {"granularity": gra, "clf": clf, "selection_type": st, "N_PCs": N_PCs}
                print("\n\n\n-----> Fold -", it)
                print("\n\n\n", params, "\n\n\n")

                model.change_hyperparams(params)

                model.fit_after_tsfresh(X_train, y_train)

                if model.one_class_:
                    Classifiers_result = "Error: Only one class"
                    break
                else:
                    y_pred = model.predict_after_tsfresh(X_test)
                    acc = balanced_accuracy_score(y_test, y_pred)

                    Classifiers_result[name]["Accuracy"] = acc
                    Classifiers_result[name]["time"] = model.predict_time_
                    Classifiers_result[name]["features"] = model.X_selected_.shape[1]
                    Classifiers_result[name]["variation"] = np.sum(model.variation_kept_)
                    Classifiers_result[name]["matrix"] = confusion_matrix(y_test, y_pred)
                    Classifiers_result[name]["y_test"] = y_test
                    Classifiers_result[name]["y_pred"] = y_pred

            with open("Classification/Classifiers_result__{}__{}__{}__.pkl".format(input_id, it, st), "wb") as f:
                pickle.dump(Classifiers_result, f)

            model.reset()
