from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import pandas as pd
import numpy as np

def import_models(model_list=None):
    """
    Import and define models to evaluate.
    Args:
        list (list): List of model names to include. If None, include all.
    Returns:
        models (dict): Dictionary of model name and instance.

    Available models:
        - "KNN": K-Nearest Neighbors
        - "RandomForest": Random Forest Classifier
        - "SVM": Support Vector Machine
        - "KNN_OVR": KNN with One-vs-Rest
        - "RF_OVR": Random Forest with One-vs-Rest
        - "SVM_OVR": SVM with One-vs-Rest
    """
    avaiable_models = {
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
    "KNN_OVR": OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15, weights="distance"), n_jobs=-1),
    "RF_OVR": OneVsRestClassifier(RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample", n_jobs=-1), n_jobs=-1),
    "SVM_OVR": OneVsRestClassifier(SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42), n_jobs=-1)
    }
    if model_list is not None:
        models = {name: model for name, model in avaiable_models.items() if name in model_list}
    print("\nModels to evaluate:", list(models.keys()))
    return models

# Prepare data for modeling: feature selection, train-test split, encoding.
def prepare_data_for_modeling(df, show_info=True):
    """
    Prepare data for modeling: feature selection, train-test split, encoding.
    Args:
        df (pd.DataFrame): The input DataFrame with features and target.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test:
        Split and encoded datasets ready for modeling.
    """
    # define your features (add all the ones you want to test).
    features = df.columns.difference(["target","random_index"])  # all except this
    X = df[features].copy()
    y = df["target"].copy()  # "MS", "Giant", "WD"

    # Encode y for metrics like multi-class ROC
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # First, we take out the test set (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # From what was left (80%), we split for validation (20% of 80% = 16% of the total).
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )

    # 1. Convert Feature arrays (X) to pandas DataFrames
    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_test_df = pd.DataFrame(X_test)

    # 2. Convert Target arrays (y) to pandas Series (or DataFrame)
    #    A Series is standard for a single target column.
    y_train_series = pd.Series(y_train)
    y_val_series = pd.Series(y_val)
    y_test_series = pd.Series(y_test)

    print("\nData prepared and split into train, val, test.")

    if show_info:
        print("\nSizes:")
        print("Train:", X_train.shape)
        print("Val:  ", X_val.shape)
        print("Test: ", X_test.shape)

    return X_train_df, X_val_df, X_test_df, y_train_series, y_val_series, y_test_series

# Run Random Forest to determine feature importances and select top-K features.
def rf_feature_selection(X_train, y_train, n_features=5, show_importance=True):
    """
    Run Random Forest to determine feature importances and select top-K features.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    Returns:
        topK (list): List of top-K feature names based on importance.
    """
    rf_base = RandomForestClassifier(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    rf_base.fit(X_train, y_train)

    importances = pd.Series(rf_base.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    if show_importance:
        print("\nImportancia (RF):", importances)

    # K = number of top features to select
    K = min(n_features, X_train.shape[1])
    topK = importances.index[:K].tolist()
    print("\nTop-K features:", topK)
    return topK

# Run Recursive Feature Elimination (RFE) with Random Forest to select top-K features.
def rfe_feature_selection(X_train, y_train, n_features=5):
    """
    Run Recursive Feature Elimination (RFE) with Random Forest to select top-K features.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        K (int): Number of features to select.
    Returns:
        selected_features (list): List of selected feature names.
    """
    rf_for_rfe = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=7, class_weight="balanced_subsample"
    )
    rfe = RFE(estimator=rf_for_rfe, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)

    selected_mask = rfe.support_
    selected_features = X_train.columns[selected_mask].tolist()
    print("\nRFE selected:", selected_features)
    return selected_features

#compare features selected by RF and RFE
def compare_feature_selection(topK_rf, topK_rfe):
    """
    Compare features selected by Random Forest and RFE.
    Args:
        topK_rf (list): Features selected by Random Forest.
        topK_rfe (list): Features selected by RFE.
    """
    set_rf = set(topK_rf)
    set_rfe = set(topK_rfe)

    common_features = set_rf.intersection(set_rfe)
    only_rf = set_rf - set_rfe
    only_rfe = set_rfe - set_rf

    print(f"Common features ({len(common_features)}): {common_features}")
    print(f"Only RF ({len(only_rf)}): {only_rf}")
    print(f"Only RFE ({len(only_rfe)}): {only_rfe}")

# Evaluate a classifier with cross-validation and test set.
def eval_model(clf, X_tr, y_tr, X_te, y_te):
    """
    Evaluate a classifier with cross-validation and test set.
    Args:
        clf: Classifier instance (must implement fit and predict).
        X_tr, y_tr: Training data.
        X_te, y_te: Test data.
    Returns:
        acc_cv, f1_cv: Cross-validated accuracy and F1-score on training data.
        acc, f1, auc: Accuracy, F1-score, and ROC-AUC on test data.
        conf_matrix: Confusion matrix on test data.
    """
    le = LabelEncoder()
    le.fit(y_tr)
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_cv = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="accuracy").mean()
    f1_cv  = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()
    # Fit + test
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro")
    # ROC-AUC one-vs-rest (requiere predict_proba)
    if hasattr(clf, "predict_proba"):
        y_bin_te = label_binarize(y_te, classes=np.arange(len(le.classes_)))
        y_proba  = clf.predict_proba(X_te)
        auc = roc_auc_score(y_bin_te, y_proba, average="macro", multi_class="ovr")
    else:
        auc = np.nan
    return acc_cv, f1_cv, acc, f1, auc, confusion_matrix(y_te, y_pred)

#Comparison of training with all features vs. K features.
def compare_model_performance(X_train, y_train, X_val, y_val, X_test, y_test, selected_features):
    """
    Compare model performance using all features vs. selected features.
    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        selected_features (list): List of selected feature names.
    """
    # Base Random Forest model
    rf_final = RandomForestClassifier(
        n_estimators=600, max_depth=None, n_jobs=-1, random_state=123, class_weight="balanced_subsample"
    )

    # (A) All features
    metrics_all = eval_model(rf_final, X_train, y_train, X_test, y_test)

    # (B) Only features selected by RFE
    X_train_sel = X_train[selected_features]
    X_test_sel  = X_test[selected_features]
    metrics_sel = eval_model(rf_final, X_train_sel, y_train, X_test_sel, y_test)

    # (C) ONLY features selected by RFE - VALIDATION
    X_val_sel = X_val[selected_features]
    metrics_sel_val = eval_model(rf_final, X_train_sel, y_train, X_val_sel, y_val)


    print("\n=== RF Results (all) ===")
    print(f"CV Acc: {metrics_all[0]:.3f} | CV Macro-F1: {metrics_all[1]:.3f} | Test Acc: {metrics_all[2]:.3f} | Test Macro-F1: {metrics_all[3]:.3f} | Test ROC-AUC: {metrics_all[4]:.3f}")
    print("Confusion matrix (test):\n", metrics_all[5])

    print("\n=== RF Results (RFE) ===")
    print(f"CV Acc: {metrics_sel[0]:.3f} | CV Macro-F1: {metrics_sel[1]:.3f} | Test Acc: {metrics_sel[2]:.3f} | Test Macro-F1: {metrics_sel[3]:.3f} | Test ROC-AUC: {metrics_sel[4]:.3f}")
    print("Confusion matrix (test):\n", metrics_sel[5])

    print("\n=== RF Results (RFE) Validation ===")
    print(f"CV Acc: {metrics_sel_val[0]:.3f} | CV Macro-F1: {metrics_sel_val[1]:.3f} | Test Acc: {metrics_sel_val[2]:.3f} | Test Macro-F1: {metrics_sel_val[3]:.3f} | Test ROC-AUC: {metrics_sel_val[4]:.3f}")
    print("Confusion matrix (test):\n", metrics_sel_val[5])

def compute_metrics(clf, X_tr, y_tr, X_ev, y_ev, class_names):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_ev)
    acc = accuracy_score(y_ev, y_pred)
    f1  = f1_score(y_ev, y_pred, average="macro")

    auc = np.nan
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_ev)
        classes_present = np.unique(y_ev)

        if len(classes_present) == 1:
            auc = np.nan  # AUC it's not defined with one class
        elif len(classes_present) == 2:
            #Binary AUC: use only the probability of the positive class
            pos_class = classes_present[1]
            pos_idx = list(clf.classes_).index(pos_class)
            y_true_bin = (y_ev == pos_class).astype(int)
            auc = roc_auc_score(y_true_bin, proba[:, pos_idx])
        else:
            # Overall multi-class AUC
            y_bin = label_binarize(y_ev, classes=clf.classes_)
            auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

    cm = confusion_matrix(y_ev, y_pred)

    #Convert class_names to strings
    string_class_names = [str(name) for name in class_names]

    report = classification_report(y_ev, y_pred, target_names=string_class_names, zero_division=0)
    return acc, f1, auc, cm, report

def compare_in_validation(models, X_train, y_train, X_val, y_val, show_classification_report=True):
    val_rows = []
    val_details = {}  # to store confusion matrices and reports
    le = LabelEncoder()
    le.fit(y_train)

    for name, clf in models.items():
        acc, f1, auc, cm, rep = compute_metrics(clf, X_train, y_train, X_val, y_val, le.classes_)
        val_rows.append({"Model": name, "Val_Accuracy": acc, "Val_MacroF1": f1, "Val_ROC_AUC": auc})
        val_details[name] = {"cm": cm, "report": rep}

    df_val = pd.DataFrame(val_rows).sort_values("Val_MacroF1", ascending=False)
    print("\n=== Validation results ===")
    print(df_val)

    best_model_name = df_val.iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"\nSelected best model on validation: {best_model_name}")
    print("\nConfusion matrix (VAL):\n", val_details[best_model_name]["cm"])
    if show_classification_report:
        print("\nClassification report (VAL):\n", val_details[best_model_name]["report"])

    return best_model_name

def retrain_best_model(X_train, y_train, X_val, y_val, X_test, y_test, best_model_name):
    le = LabelEncoder()
    le.fit(y_train)

    best_model = import_models([best_model_name])
    best_model = best_model[best_model_name]

    X_trfin = np.vstack([X_train, X_val])
    y_trfin = np.hstack([y_train, y_val])

    test_acc, test_f1, test_auc, test_cm, test_rep = compute_metrics(
        best_model, X_trfin, y_trfin, X_test, y_test, le.classes_
    )

    print("\n=== Test results (best model) ===")
    print(f"Model: {best_model_name}")
    print(f"Test Accuracy:  {test_acc:.3f}")
    print(f"Test Macro-F1:  {test_f1:.3f}")
    print(f"Test ROC-AUC:   {test_auc:.3f}" if not np.isnan(test_auc) else "Test ROC-AUC:   NA")
    print("\nConfusion matrix (TEST):\n", test_cm)
    print("\nClassification report (TEST):\n", test_rep)