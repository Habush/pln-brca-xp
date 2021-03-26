__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, confusion_matrix, average_precision_score, plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import TransformerMixin

params = {
          # 'n_estimators': [300, 400, 500, 600, 700],
          'n_estimators': [20, 50, 80, 120, 150],
          'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [3, 4, 5, 6],
          'subsample': [0.6, 0.8, 1.0],
          'colsample_bytree': [0.6, 0.8, 1.0],
          'min_child_weight': [1, 2, 3, 4, 5],
          'scale_pos_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
          'max_delta_step': [1, 2, 3, 4, 5]
}

params_no_scale = {
          # 'n_estimators': [300, 400, 500, 600, 700],
          'n_estimators': [20, 50, 80, 120, 150],
          'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07],
          'gamma': [0.5, 1, 1.5, 2, 5],
          'max_depth': [3, 4, 5, 6],
          'subsample': [0.6, 0.8, 1.0],
          'colsample_bytree': [0.6, 0.8, 1.0],
          'min_child_weight': [1, 2, 3, 4, 5]
          # 'scale_pos_weight': [0.1, 0.3, 0.5, 0.7, 0.9],
          # 'max_delta_step': [1, 2, 3, 4, 5]
}

seed = 42
st_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

score_cols = ["test_balanced_accuracy", "test_recall_0", "test_precision_0", "test_recall_1",
              "test_precision_1", "test_auc", "test_specificity", "test_average_precision_0"]
cv_df_cols = ["balanced_accuracy", "recall_0", "precision_0", "recall_1", "precision_1", "auc", "specificity", "average_precision_0"]


def calc_scores(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    recall_0, recall_1 = recall_score(y_test, y_pred, pos_label=0), recall_score(y_test, y_pred, pos_label=1)
    precision_0, precision_1 = precision_score(y_test, y_pred, pos_label=0), precision_score(y_test, y_pred, pos_label=1)
    sp = specificity(y_test, y_pred)
    acc = balanced_accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    av_0 = average_precision_score(y_test, y_pred, pos_label=0)
    arr = np.array([[acc, recall_0, precision_0, recall_1, precision_1, auc_score, sp, av_0]])
    return pd.DataFrame(data=arr, columns=cv_df_cols)


def recall_0(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)


def precision_0(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=0)


def auc_0(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred[:,0])
    return auc(fpr, tpr)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    return spec

def avp_0(y_true, y_pred):
    avp = average_precision_score(y_true, y_pred, pos_label=0)
    return avp

average_precision_0 = make_scorer(avp_0, greater_is_better=True)
average_precision_1 = make_scorer(average_precision_score, greater_is_better=True)
auc_cl_0 = make_scorer(auc_0, greater_is_better=True, needs_proba=True)
spec_score = make_scorer(specificity, greater_is_better=True)
prec_0 = make_scorer(precision_0, greater_is_better=True)
scoring = {"balanced_accuracy": make_scorer(balanced_accuracy_score),
           "recall_0": make_scorer(recall_0), "precision_0": make_scorer(precision_0),
           "recall_1": make_scorer(recall_score), "precision_1": make_scorer(precision_score), "auc": "roc_auc",
           "specificity": spec_score, "average_precision_0": average_precision_0}


# cross_validation

def print_score_comparison(raw_score, emb_score, target_feature="posOutcome",
                           header_1="Raw Score", header_2="Embedding Score", title="Validation set"):
    print("\t{0}\t{1}\n\t\t\t{2}\t\t{3}".format(title, target_feature, header_1, header_2))
    print("\t\t-------------------------------------------------------")
    print("balanced_accuracy:\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["balanced_accuracy"].mean(),
                                                                emb_score["balanced_accuracy"].mean()))
    print("recall_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["recall_0"].mean(), emb_score["recall_0"].mean()))
    print("precision_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["precision_0"].mean(),
                                                            emb_score["precision_0"].mean()))
    print("recall_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["recall_1"].mean(), emb_score["recall_1"].mean()))
    print("precision_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["precision_1"].mean(),
                                                            emb_score["precision_1"].mean()))
    print("auc:\t\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(raw_score["auc"].mean(), emb_score["auc"].mean()))


def print_score_comparison_v2(cv_scores_1, cv_scores_2, test_scores_1, test_scores_2,
                           header_1="Raw Score", header_2="Embedding Score", opt="Balanced Opt"):
    print("\t{0} - {1}\n\n\t\t\t{2}\t\t\t\t{3}".format("Validation set", opt, header_1, header_2))
    print("\t\t-------------------------------------------------------")
    print("balanced_accuracy:\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["balanced_accuracy"].mean(),
                                                                cv_scores_2["balanced_accuracy"].mean()))
    print("recall_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["recall_0"].mean(), cv_scores_2["recall_0"].mean()))
    print("precision_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["precision_0"].mean(),
                                                            cv_scores_2["precision_0"].mean()))
    print("recall_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["recall_1"].mean(), cv_scores_2["recall_1"].mean()))
    print("precision_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["precision_1"].mean(),
                                                            cv_scores_2["precision_1"].mean()))
    print("auc:\t\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(cv_scores_1["auc"].mean(), cv_scores_2["auc"].mean()))

    print("\n")
    print("\t{0} - {1}\n\n\t\t\t{2}\t\t\t\t{3}".format("Test set", opt, header_1, header_2))
    print("\t\t-------------------------------------------------------")
    print("balanced_accuracy:\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["balanced_accuracy"].mean(),
                                                                test_scores_2["balanced_accuracy"].mean()))
    print("recall_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["recall_0"].mean(), test_scores_2["recall_0"].mean()))
    print("precision_0:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["precision_0"].mean(),
                                                            test_scores_2["precision_0"].mean()))
    print(
        "recall_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["recall_1"].mean(), test_scores_2["recall_1"].mean()))
    print("precision_1:\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["precision_1"].mean(),
                                                            test_scores_2["precision_1"].mean()))
    print("auc:\t\t\t{0:.2%}\t\t\t\t{1:.2%}\n".format(test_scores_1["auc"].mean(), test_scores_2["auc"].mean()))

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time

    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def param_tuning(X, y, n_folds=5, param_comb=25, scoring='roc_auc', jobs=12, scale_weight=True):
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                        silent=True, nthread=1)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    if scale_weight:
        rand_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=scoring, n_jobs=jobs,
                                         cv=skf.split(X, y), verbose=3, random_state=42)
    else:
        rand_search = RandomizedSearchCV(xgb, param_distributions=params_no_scale, n_iter=param_comb, scoring=scoring,
                                         n_jobs=jobs,
                                         cv=skf.split(X, y), verbose=3, random_state=42)
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    rand_search.fit(X, y)
    timer(start_time)
    print("Best Score: {:.3%}".format(rand_search.best_score_))
    print(rand_search.best_params_)
    return rand_search


def get_scores(cv_results, score_keys=None, df_cols=None):
    if score_keys is None:
        score_keys = score_cols
    if df_cols is None:
        score_keys = score_cols
    scores = np.empty([1, len(score_keys)])
    for i, s in enumerate(score_keys):
        scores[0][i] = np.mean(cv_results[s])
    scores_df = pd.DataFrame(data=scores, columns=cv_df_cols)
    return scores_df


def evaluate_embedding(path, outcome_df, target="posOutcome", merge_col="patient_ID", n_jobs=-1):
    emb_df = pd.read_csv(path, sep="\t")
    emb_outcome_df = pd.merge(outcome_df, emb_df, on=merge_col)
    X_emb, y_emb = emb_outcome_df[emb_outcome_df.columns.difference([merge_col, target])], emb_outcome_df[target]
    X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(X_emb, y_emb, test_size=0.3, random_state=seed)
    rand_search_emb = param_tuning(X_train_emb, y_train_emb, jobs=n_jobs)
    params = rand_search_emb.best_params_
    clf_emb = rand_search_emb.best_estimator_
    cv_res = cross_validate(clf_emb, X_train_emb, y_train_emb, scoring=scoring, n_jobs=n_jobs, verbose=1,
                            return_train_score=True,
                            cv=st_cv)
    cv_res_df = get_scores(cv_res)
    clf_emb.fit(X_train_emb, y_train_emb)
    test_scores_df = calc_scores(clf_emb, X_test_emb, y_test_emb)

    return params, cv_res_df, test_scores_df


def load_features(path):
    feats = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            feats.append(line.strip())

    return feats


def evaluate_ge(df, target="posOutcome", outcome_cols=None, feats=None, jobs=-1,
                params=None, scoring=scoring, rand_scoring="roc_auc", custom_clf=None, scale_weight=True, stratify=True, split=True):

    if split:
        X, y = df.drop([target], axis=1), df[target]
        if stratify:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    else:
        x_train, x_test, y_train, y_test = df
    if feats is not None or outcome_cols is not None:
        if feats is not None and outcome_cols is not None:
            cols = outcome_cols + feats
        elif feats is not None and outcome_cols is None:
            cols = feats
        else:
            cols = outcome_cols
        x_train = x_train[cols]
        x_test = x_test[cols]

    if params is None:
        rand_search = param_tuning(x_train, y_train, scoring=rand_scoring, jobs=jobs, scale_weight=scale_weight)
        clf_params = rand_search.best_params_
    else:
        clf_params = params

    if custom_clf is None:
        clf = XGBClassifier(**clf_params)
    else:
        clf = custom_clf
    cv_res = cross_validate(clf, x_train, y_train, scoring=scoring, cv=st_cv, n_jobs=jobs)

    cv_res_df = get_scores(cv_res)
    clf.fit(x_train, y_train)
    test_scores_df = calc_scores(clf, x_test, y_test)

    print("====================================================")
    print("CV Score: \n{0}\n".format(cv_res_df.mean()))
    print("====================================================")
    print("Test Score:\n{0}\n".format(test_scores_df.mean()))
    if params is None:
        return clf_params, clf, cv_res_df, test_scores_df
    else:
        return clf, cv_res_df, test_scores_df


def mean_discretize_dataset(X, bins_labels=None):
    if bins_labels is None:
        bins_labels = [-1, 0, 1]
    features = X.columns.to_list()
    bin_dict = {}
    X_disc = X
    for ft in features:
        r1 = X[ft].mean() - X[ft].std() / 2
        r2 = X[ft].mean() + X[ft].std() / 2
        bin_dict[ft] = [-np.inf, r1, r2, np.inf]
    le = LabelEncoder()

    le.fit(bins_labels)

    for ft in bin_dict:
        X_disc[ft] = le.transform(pd.cut(X_disc[ft], bins=bin_dict[ft], labels=bins_labels))

    ohe = OneHotEncoder(handle_unknown="ignore")
    transformed = ohe.fit_transform(X_disc).toarray()
    X_disc = pd.DataFrame(transformed, columns=ohe.get_feature_names(features))
    return X_disc

def median_discretize_dataset(X):
    X_disc = X - X.median()
    return X_disc

def optimize_param(params, opt_param, val_range, df, target="posOutcome" ,step=1, feats=None, outcome_cols=None, metric=None, n_jobs=-1, stratify=True):
    errors = []
    X, y = df.drop([target], axis=1), df[target]
    if stratify:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    if feats is not None:
        if outcome_cols is not None:
            cols = outcome_cols + feats
        else:
            cols = feats
        X_train = X_train[cols]
    for k in np.arange(val_range[0], val_range[1], step):
        params[opt_param] = k
        clf = XGBClassifier(**params)
        cv_res = cross_validate(clf, X_train, y_train, scoring=metric, n_jobs=n_jobs, verbose=1,
                                cv=st_cv)
        print(cv_res)
        score = np.mean(cv_res["test_score"])
        if metric is None:
            errors.append({'K': k, 'LogLoss': score})

        else:
            errors.append({'K': k, 'Score': score})

    if metric is None:
        least_err = sorted(errors, key=lambda k: k["LogLoss"])[0]
        print("Got least LogLoss {0:.2f} at n={1}".format(least_err["LogLoss"],
                                                          least_err["K"]))
        print(errors)
        return least_err["K"]
    else:
        highest_score = sorted(errors, key=lambda k: k["Score"], reverse=True)[0]
        print("Got highest score {0:.2f} at n={1}".format(highest_score["Score"], highest_score["K"]))
        print(errors)
        return highest_score["K"]

def optimize_k_v1(df, target, exclude=None):
    if exclude is None:
        exclude = ["patient_ID"]
    df = df.drop(exclude, axis=1)
    data = df.to_numpy()
    errors = []
    for k in range(1, 20, 2):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(data)
        df_imputed = pd.DataFrame(imputed, columns=df.columns)

        X = df_imputed.drop(target, axis=1)
        y = df_imputed[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append({'K': k, 'RMSE': error})

    return errors


from sklearn.metrics import mean_squared_error as rmse


def optimize_k_v2(X, y):
    errors = []
    for k in range(1, 20, 2):
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(X)
        df_imputed = pd.DataFrame(imputed, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(df_imputed, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train.values, y_train.values)
        preds = model.predict(X_test.values)
        error = log_loss(y_test.values, preds)
        errors.append({'K': k, 'Logloss': error})

    least_err = sorted(errors, key=lambda k: k["Logloss"])[0]
    print("Got least Logloss {0:.2f} at n={1}".format(least_err["Logloss"],
                                                   least_err["K"]))

    return least_err["K"]


def impute_dataset_v1(df, imputer, target="posOutcome",
                      drop_target=True):
    if drop_target:
        X = df.drop([target], axis=1)
    else:
        X = df
    X_new = imputer.fit_transform(X)
    df_imputed = pd.DataFrame(X_new, columns=X.columns, index=X.index)
    print("Shape imputed: " + str(df_imputed.shape))
    if drop_target:
        p_outcome_df = df["posOutcome"]
        df_imputed = pd.concat([p_outcome_df, df_imputed], axis=1,
                               verify_integrity=True)
        print("Shape imputed concat: " + str(df_imputed.shape))
    return df_imputed


def impute_knn(X, y):
    n = optimize_k_v2(X, y)
    imputer = KNNImputer(n_neighbors=n)
    X_new = imputer.fit_transform(X)
    df_imputed = pd.DataFrame(X_new, columns=X.columns, index=X.index)
    return df_imputed


def one_hot_encode(df, cat_features):
    X_cats = df[cat_features]
    ohe = OneHotEncoder(dtype=np.int64)
    X_ohe = ohe.fit_transform(X_cats).toarray()
    fts_names = ohe.get_feature_names(cat_features)
    ohe_df = pd.DataFrame(X_ohe, columns=fts_names, index=X_cats.index)
    non_cat_df = df[df.columns.difference(cat_features)]
    df_encoded = pd.concat([non_cat_df, ohe_df], axis=1, verify_integrity=True)
    return df_encoded


def find_diff(df1, df2, index="patient_ID"):
    def highlight_diff(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        other = data.xs('First', axis='columns', level=-1)
        return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
                            index=data.index, columns=data.columns)

    df_all = pd.concat([df1.set_index(index), df2.set_index(index)],
                       axis='columns', keys=['First', 'Second'])
    df_final = df_all.swaplevel(axis='columns')[df1.columns[1:]]
    df_final.style.apply(highlight_diff, axis=None)
    return df_final


def plot_percentages(df, col="posOutcome"):
    x = df[col].value_counts().sort_values().plot(kind="barh")
    totals = []
    for i in x.patches:
        totals.append(i.get_width())
    total = sum(totals)
    for i in x.patches:
        x.text(i.get_width() + .3, i.get_y() + .20,
               str(round((i.get_width() / total) * 100, 2)) + '%',
               fontsize=10, color='black')
    x.grid(axis="x")
    plt.suptitle(col, fontsize=20)
    plt.show()


def label_encode(df):
    # label encode each non-null value [Fixes the tumor -1 issue]
    df = df.apply(lambda col: pd.Series(
        LabelEncoder().fit_transform(col[col.notnull()]),
        index=col[col.notnull()].index))
    # Change the dataframe features to int
    # Note using "Int64" instead of np.int64 because NaN values
    # can't be converted to int by default
    # See https://stackoverflow.com/q/11548005/3380414
    for col in df.columns:
        df[col] = df[col].astype("Int64")

    return df

def convert_df_to_atomese_repr(df, feats=None):
    if feats is not None:
        df_in = df[feats]
    else:
        df_in = df

    overexpr_col_names = {}
    underexpr_col_names = {}

    def overexpr_series(col):
        filtered = np.where(col > 0, col, 0)
        name = col.name + "_overexpr"
        overexpr_col_names[col.name] = name
        return pd.Series(filtered, name=name, index=col.index)

    def underexpr_series(col):
        filtered = np.where(col < 0, np.abs(col), 0)
        name = col.name + "_underexpr"
        underexpr_col_names[col.name] = name
        return pd.Series(filtered, name=name, index=col.index)

    overexpr_df = df_in.apply(overexpr_series)
    overexpr_df = overexpr_df.rename(columns=overexpr_col_names)

    underexpr_df = df_in.apply(underexpr_series)
    underexpr_df = underexpr_df.rename(columns=underexpr_col_names)

    df_out = pd.concat([overexpr_df, underexpr_df], axis=1, join="inner", verify_integrity=True)
    return df_out


def get_train_test_set(df, train_ids, test_ids):
    train_set = []
    test_set = []
    with open(train_ids, "r") as fp:
        for line in fp.readlines():
            train_set.append(int(line.strip()))
    with open(test_ids, "r") as fp:
        for line in fp.readlines():
            test_set.append(int(line.strip()))

    train_idx, test_idx = pd.Index(train_set), pd.Index(test_set)
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]
    X_train, y_train = df_train.drop(["posOutcome"], axis=1), df_train["posOutcome"]
    X_test, y_test = df_test.drop(["posOutcome"], axis=1), df_test["posOutcome"]
    return X_train, X_test, y_train, y_test


def alt_train_test(df, features, target="posOutcome"):
    cv_score_matrix = np.empty((3, 8))
    test_score_matrix = np.empty((3, 8))
    X, y = df.drop([target], axis=1), df[target]
    for i in range(0, 3):
        seed = int(np.random.random() * 100)
        print("Using seed: %d" % seed)
        x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y, test_size=0.3)
        _, _, cv_scores, test_scores = evaluate_ge((x_train, x_test, y_train, y_test), split=False, rand_scoring=average_precision_0, feats=features)
        cv_score_matrix[i:,] = cv_scores.mean()
        test_score_matrix[i:,] = test_scores.mean()

    cv_scores_df = pd.DataFrame(cv_score_matrix, columns=cv_df_cols)
    test_scores_df = pd.DataFrame(test_score_matrix, columns=cv_df_cols)
    return cv_scores_df, test_scores_df


class MQNormalizer(TransformerMixin):
    """
    Transforms raw gene expressions to a median and quantile-normalized version
    """
    def __init__(self, n_quantiles=5, subsample=int(1e5), scaler=None):
        self.medians = None
        if scaler is None:
            self.qnorm = QuantileTransformer(n_quantiles, subsample=subsample)
        else:
            self.qnorm = scaler

    def fit_transform(self, X, **kwargs):
        self.medians = X.median()
        X_med = X - self.medians
        X_norm = convert_df_to_atomese_repr(X_med)
        X_q = self.qnorm.fit_transform(X_norm)
        X_q = pd.DataFrame(X_q, columns=X_norm.columns, index=X_norm.index)
        return X_q

    def transform(self, X):
        X_med = X - self.medians
        X_norm = convert_df_to_atomese_repr(X_med)
        X_q = self.qnorm.transform(X_norm)
        X_q = pd.DataFrame(X_q, columns=X_norm.columns, index=X_norm.index)
        return X_q

def embedding_effect(X_train_raw, X_train_emb, X_test_raw, X_test_emb, y_train, y_test, num_rounds=20, scoring="balanced_accuracy"):

    train_scores = {}
    test_scores = {}
    params = []
    for i in range(num_rounds):
        print("Adding first %d dims from embedding: " % i)
        if i == 0: #Just use raw
            X_train, X_test = X_train_raw, X_test_raw
        else:
            X_train, X_test = pd.concat([X_train_raw, X_train_emb.iloc[:, :i]], axis=1, verify_integrity=True), pd.concat([X_test_raw, X_test_emb.iloc[:, :i]], axis=1, verify_integrity=True)

        params_raw_emb, clf_raw_emb, cv_scores_raw_emb, test_scores_raw_emb = evaluate_ge((X_train, X_test, y_train, y_test), split=False, rand_scoring=scoring)
        for score in cv_df_cols:
            if score not in train_scores:
                train_scores[score] = []
                test_scores[score] = []
            train_scores[score].append(cv_scores_raw_emb[score].mean())
            test_scores[score].append(test_scores_raw_emb[score].mean())
            params.append(params_raw_emb)
    return params, train_scores, test_scores