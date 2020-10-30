# Title: Assignment #2
# Author: Brendan Teasdale
# Course: Machine Learning (CSCI 444)
# Professor: Jacob Levman
# Version: 0.0.1

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import loadmat
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def loadDigitDataset(file: str) -> dict:
    dataset = loadmat(Path(__file__).resolve().parent / file)
    return dataset


def loadCancerDataset(file: str) -> pd.DataFrame:
    dataset = pd.read_csv(file)
    return cleanData(dataset=dataset)


# Performed when dataset is loaded.
# Preprocess data
def cleanData(dataset: Union[pd.DataFrame, dict]) -> pd.DataFrame:
    # Need to remove empty, unnamed column in dataset
    # Implentation from https://stackoverflow.com/a/43983654
    dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]
    dataset = dataset.drop(columns=["id"])
    return dataset


# Only grab 8's and 9's from larger dataset
def filterDataset(x_data: pd.DataFrame, x_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    idx_8_9 = (np.ravel(x_labels) == 8) | (np.ravel(x_labels) == 9)
    return x_data[:, :, idx_8_9], x_labels[:, idx_8_9]


def visualizeDigit(x_data: pd.DataFrame, x_labels: np.ndarray) -> None:
    index = np.random.randint(low=0, high=x_labels.shape[-1])
    img = x_data[:, :, index]
    plt.imshow(img, cmap="Greys")
    plt.title("Handwritten digit: " + str(x_labels[:, index]) + " \n@ index: " + str(index))
    plt.savefig(fname="digit.png")
    plt.close()


# Reshape data to fit size requirements for classifiers
def reshapeData(x_data: pd.DataFrame, x_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    x_data = np.transpose(x_data, [2, 0, 1])
    x_data = x_data.reshape(x_data.shape[0], np.prod(x_data.shape[1:]))
    x_labels = np.ravel(np.transpose(x_labels, [1, 0]))
    return x_data, x_labels


def getScoreAverage(cv: dict) -> float:
    scores = np.array(cv["test_score"])
    return np.average(scores)


def getValidationMetrics(x_data: pd.DataFrame, x_labels: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    svm_linear = SVC(kernel="linear", gamma="scale")
    svm_rbf = SVC(kernel="rbf", gamma="scale")
    rf_classifier = RandomForestClassifier(n_estimators=100)
    knn_classifier_1 = KNeighborsClassifier(n_neighbors=1, metric="minkowski", p=2, algorithm="auto")
    knn_classifier_5 = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2, algorithm="auto")
    knn_classifier_10 = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2, algorithm="auto")

    # we will pass StratifiedKFold in as our iterable for cross_validate's cv parameter
    skf_val = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    svm_linear_cv = cross_validate(svm_linear, x_data, x_labels, n_jobs=-1, cv=skf_val)
    svm_rbf_cv = cross_validate(svm_rbf, x_data, x_labels, n_jobs=-1, cv=skf_val)
    rf_classifier_cv = cross_validate(rf_classifier, x_data, x_labels, n_jobs=-1, cv=skf_val)
    knn_classifier_1_cv = cross_validate(knn_classifier_1, x_data, x_labels, n_jobs=-1, cv=skf_val)
    knn_classifier_5_cv = cross_validate(knn_classifier_5, x_data, x_labels, n_jobs=-1, cv=skf_val)
    knn_classifier_10_cv = cross_validate(knn_classifier_10, x_data, x_labels, n_jobs=-1, cv=skf_val)

    svm_lin_average_score = getScoreAverage(svm_linear_cv)
    svm_rbf_average_score = getScoreAverage(svm_rbf_cv)
    rf_average_score = getScoreAverage(rf_classifier_cv)
    knn_1_average_score = getScoreAverage(knn_classifier_1_cv)
    knn_5_average_score = getScoreAverage(knn_classifier_5_cv)
    knn_10_average_score = getScoreAverage(knn_classifier_10_cv)
    return (
        svm_lin_average_score,
        svm_rbf_average_score,
        rf_average_score,
        knn_1_average_score,
        knn_5_average_score,
        knn_10_average_score,
    )


def printScores(
    question: int,
    svm_lin_score: float,
    svm_rbf_score: float,
    rf_score: float,
    knn1_score: float,
    knn5_score: float,
    knn10_score: float,
) -> None:
    print("\nScores for question: " + str(question))
    print("svm_linear score: " + str(svm_lin_score))
    print("svm_rbf score: " + str(svm_rbf_score))
    print("rf score: " + str(rf_score))
    print("knn_1 score: " + str(knn1_score))
    print("knn_5 score: " + str(knn5_score))
    print("knn_10 score: " + str(knn10_score))


def applyFeatureScaling(x_data: pd.DataFrame) -> pd.DataFrame:
    sc = StandardScaler()
    return sc.fit_transform(x_data)


def labelEncodeData(x_labels: np.ndarray) -> np.ndarray:
    return LabelEncoder().fit_transform(x_labels)


def computeAUC(dataset: pd.DataFrame, targets: np.ndarray) -> list:
    aucs = []
    for i in range(0, dataset.shape[-1]):
        data = dataset.iloc[:, i].values
        group1 = data[targets == 0]
        group2 = data[targets == 1]
        auc = mannwhitneyu(group1, group2).statistic / (len(group1) * len(group2))
        aucs.append(auc)
    return aucs


def sortAUCS(aucs: np.array) -> list:
    sort_aucs = []
    for auc in aucs:
        sort_aucs.append(abs(auc - 0.5))
    # Implementation from https://www.geeksforgeeks.org/python-indices-of-n-largest-elements-in-list/
    n_max_indicies = sorted(range(len(sort_aucs)), key=lambda sub: sort_aucs[sub])[-10:]
    return n_max_indicies


def AUCToMarkdown(dataset: pd.DataFrame, aucs: np.ndarray) -> None:
    # Flip array to be in ascending order
    sorted_aucs = np.array(np.flip(sortAUCS(aucs)))
    feat_names = dataset.columns[sorted_aucs]
    cols = ["Features", "AUC"]
    auc_dataframe = pd.DataFrame(
        index=feat_names,
        columns=cols,
        data=np.zeros([len(feat_names), len(cols)]),
    )
    for i, feat_name in enumerate(feat_names):
        auc_dataframe.loc[feat_names[i], "AUC"] = aucs[sorted_aucs[i]]
    auc_dataframe["Features"] = feat_names
    # Visualize kfold_scores dataframe as a markdown table
    print(auc_dataframe.round(4).to_markdown(index=False))
    auc_dataframe.to_json(Path(__file__).resolve().parent / "aucs.json")


def createKFoldDataframe(err_rates: list) -> pd.DataFrame:
    cols = ["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"]
    rows = ["err"]
    kfold_scores = pd.DataFrame(index=rows, columns=cols, data=np.zeros([len(rows), len(cols)]))
    kfold_scores.loc["err"] = err_rates
    # Visualize kfold_scores dataframe as a markdown table
    print(kfold_scores.round(6).to_markdown())
    return kfold_scores


def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:
    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    if np.min(df.values) < 0 or np.max(df.values) > 0.10:
        raise ValueError(
            "Your K-Fold error rates are too extreme. Ensure they are the raw error rates,\r\n"
            "and NOT percentage error rates. Also ensure your DataFrame contains error rates,\r\n"
            "and not accuracies. If you are sure you have not made either of the above mistakes,\r\n"
            "there is probably something else wrong with your code. Contact the TA for help.\r\n"
        )

    if df.loc["err", "svm_linear"] > 0.07:
        raise ValueError("Your svm_linear error rate is too high. There is likely an error in your code.")
    if df.loc["err", "svm_rbf"] > 0.03:
        raise ValueError("Your svm_rbf error rate is too high. There is likely an error in your code.")
    if df.loc["err", "rf"] > 0.05:
        raise ValueError("Your Random Forest error rate is too high. There is likely an error in your code.")
    if df.loc["err", ["knn1", "knn5", "knn10"]].min() > 0.04:
        raise ValueError("One of your KNN error rates is too high. There is likely an error in your code.")

    outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")


def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
    COLS = sorted(["svm_linear", "svm_rbf", "rf", "knn1", "knn5", "knn10"])
    df = kfold_scores
    if not isinstance(df, DataFrame):
        raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
    if kfold_scores.shape != (1, 6):
        raise ValueError("DataFrame must have 1 row and 6 columns.")
    if not np.alltrue(sorted(df.columns) == COLS):
        raise ValueError("Columns are incorrectly named.")
    if not df.index.values[0] == "err":
        raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

    outfile = Path(__file__).resolve().parent / "kfold_data.json"
    df.to_json(outfile)
    print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")


def question1() -> None:
    # Title: Using Support Vector Machines (SVM) to predict handwritten digits (8 or 9)
    # Required Input: NumberRecognitionBigger.mat

    # dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
    dataset = loadDigitDataset(file="NumberRecognitionBigger.mat")
    img_data, img_labels = dataset["X"], dataset["y"]

    # We only want digits 8 or 9
    img_data, img_labels = filterDataset(x_data=img_data, x_labels=img_labels)
    visualizeDigit(x_data=img_data, x_labels=img_labels)
    # Reshape data to have correct size for classifiers
    img_data, img_labels = reshapeData(x_data=img_data, x_labels=img_labels)

    (
        svm_lin_score,
        svm_rbf_score,
        rf_score,
        knn1_score,
        knn5_score,
        knn10_score,
    ) = getValidationMetrics(x_data=img_data, x_labels=img_labels)
    metrics_arr = np.array([svm_lin_score, svm_rbf_score, rf_score, knn1_score, knn5_score, knn10_score])
    error_rates = np.apply_along_axis(arr=metrics_arr, axis=0, func1d=lambda idx: 1 - idx)

    kfold_scores = createKFoldDataframe(error_rates)
    save_mnist_kfold(kfold_scores)
    printScores(1, svm_lin_score, svm_rbf_score, rf_score, knn1_score, knn5_score, knn10_score)


def question2() -> None:
    # Title: Description of selected dataset
    # Required Input: breast-cancer-data.csv

    dataset = loadCancerDataset(file="breast-cancer-data.csv")
    x = np.ravel(dataset.filter(["diagnosis"]))
    dataset = dataset.drop(columns=["diagnosis"])
    # B (Benign) -> 0, M (Malignant) -> 1
    x = labelEncodeData(x)
    aucs = np.array(computeAUC(dataset=dataset, targets=x))
    AUCToMarkdown(dataset=dataset, aucs=aucs)

    print("\nCount of Malignant (Group of Interest) Samples: " + str(list(x).count(1)))
    print("Count of Benign (Group not of Interest) Samples: " + str(list(x).count(0)))


def question3() -> None:
    # Title: Using Support Vector Machines (SVM) to predict ...
    # Required Input: breast-cancer-data.csv

    dataset = loadCancerDataset(file="breast-cancer-data.csv")

    # we start at index 1 to ignore 'diagnosis' column (0)
    x = dataset.iloc[:, 1 : len(dataset)].values
    y = np.ravel(dataset.filter(["diagnosis"]))
    x = applyFeatureScaling(x_data=x)
    y = labelEncodeData(x_labels=y)

    (
        svm_lin_score,
        svm_rbf_score,
        rf_score,
        knn1_score,
        knn5_score,
        knn10_score,
    ) = getValidationMetrics(x_data=x, x_labels=y)
    metrics_arr = np.array([svm_lin_score, svm_rbf_score, rf_score, knn1_score, knn5_score, knn10_score])
    error_rates = np.apply_along_axis(arr=metrics_arr, axis=0, func1d=lambda idx: 1 - idx)

    kfold_scores = createKFoldDataframe(error_rates)
    save_data_kfold(kfold_scores)

    # first parameter 3 specifies the question from which the scores correspond to
    printScores(3, svm_lin_score, svm_rbf_score, rf_score, knn1_score, knn5_score, knn10_score)


if __name__ == "__main__":
    question1()
    # question2()
    # question3()
