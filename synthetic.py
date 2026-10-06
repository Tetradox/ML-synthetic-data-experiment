import sys
from statistics import mean
import random
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, KFold
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer
from sklearn.neighbors import NearestNeighbors

SEED = 42
SAMPLE_WEIGHTS = [2, 4, 6, 8]
SYNTHETIC_SIZES = [10, 25, 50, 100]
MINORITY_THRESHOLD = 15

if len(sys.argv) < 2:
    sys.exit("Usage: py ml_testing.py filename.csv")
filename = sys.argv[1]

parameters_boosting = {"n_estimators": [50, 100, 150],
            "learning_rate": [0.05, 0.1, 0.15],
            "max_depth": [1, 2, 3],
            "subsample": [0.25, 0.50, 0.75]
            }

parameters_bagging = {"n_estimators": [50, 100, 150],
                       "max_depth": [1, 2, 3],
                       "min_samples_leaf": [1, 2, 3],
                       "min_samples_split": [2, 3, 4]}

input_data = ["ID_IG", "Time", "Temperature", "BET"]
output_data = "Selectivity"

def main() -> None:
    dataset = load_data(filename)
    models = model_creation()
    results = pipeline(models, dataset, input_data, output_data)
    print(results)


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


def model_creation() -> dict:
    return {"gboost": [GradientBoostingRegressor(random_state=SEED), parameters_boosting | {"min_samples_leaf": [1, 2, 3]}],
            "xgboost": [XGBRegressor(random_state=SEED), parameters_boosting],
            "rforest": [RandomForestRegressor(random_state=SEED, n_jobs=-1), parameters_bagging],
            "etrees": [ExtraTreesRegressor(random_state=SEED, n_jobs=-1), parameters_bagging]
            }


def create_synthesizers(metadata: Metadata) -> dict:
    return {"copula": GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True),
            "ctgan": CTGANSynthesizer(metadata, enforce_min_max_values=True),
            "tvae": TVAESynthesizer(metadata, enforce_min_max_values=True),
            "smote": None
            }


def data_division(baseline: pd.DataFrame, synthetic: list) -> dict:
    mixed_dataset = [pd.concat([baseline, synth_df], axis=0, ignore_index=True) for synth_df in synthetic]
    return {
        "synthetic_plus_baseline": mixed_dataset,
        "weighted_baseline_plus_synthetic": mixed_dataset,
        "synthetic_only": synthetic
    }


def return_metrics(cv: dict) -> list:
    return_index = np.nanargmax(cv["mean_test_r2"])
    return [
            cv["mean_test_r2"][return_index],
            -cv["mean_test_neg_root_mean_squared_error"][return_index],
            -cv["mean_test_neg_mean_absolute_error"][return_index]
            ]


def baseline_evaluation(models: dict, baseline: pd.DataFrame, train: list, test: list) -> dict:
    results = dict()
    for model in models:
        search = RandomizedSearchCV(models[model][0], models[model][1], cv=[(train, test)], n_iter=10,
                                    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"], n_jobs=-1, refit="r2")
        cv = search.fit(baseline[input_data], baseline[output_data]).cv_results_
        results[f"{model}, baseline"] = return_metrics(cv)
    return results


def synthetic_evaluation(models: dict, datasets: dict, baseline: pd.DataFrame, synthesizer_name: str, test: list, model) -> dict:
    results = dict()
    for dataset_name in datasets.keys():
        df_list = datasets[dataset_name]
        for df in df_list:
            synthetic_size = len(df) if dataset_name == "synthetic_only" else len(df) - len(baseline)
            search = RandomizedSearchCV(models[model][0], models[model][1], n_iter=10, scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"], n_jobs=-1, refit="r2")

            if dataset_name == "synthetic_plus_baseline":
                train_indices = df[input_data].drop(test).index
                search.cv = [(train_indices, test)]
                cv = search.fit(df[input_data], df[output_data]).cv_results_
                results[f"{model, dataset_name, synthesizer_name, synthetic_size}"] = return_metrics(cv)

            elif dataset_name == "synthetic_only":
                train_indices = df.index
                df = pd.concat((df, baseline.iloc[test]), axis=0, ignore_index=True)
                search.cv = [(train_indices, [index for index in df.index if index not in train_indices])]
                cv = search.fit(df[input_data], df[output_data]).cv_results_
                results[f"{model, dataset_name, synthesizer_name, synthetic_size}"] = return_metrics(cv)

            elif dataset_name == "weighted_baseline_plus_synthetic":
                for weight in SAMPLE_WEIGHTS:
                    train_indices = df[input_data].drop(test).index
                    search.cv = [(train_indices, test)]
                    weights =  [weight] * len(baseline) + [1] * (len(df) - len(baseline))
                    cv = search.fit(df[input_data], df[output_data], sample_weight=weights).cv_results_
                    results[f"{model, dataset_name, synthesizer_name, synthetic_size, weight}"] = return_metrics(cv)
    return results


def smote_synthetic_generation(baseline: pd.DataFrame, train: list) -> dict:
    results = list()
    minority_class = baseline.iloc[train][baseline[output_data] <= MINORITY_THRESHOLD].reset_index(drop=True).to_numpy()
    if len(minority_class) == 0:
        return [pd.DataFrame(columns=["ID_IG", "Time", "Temperature", "BET", "Selectivity"]) for _ in SYNTHETIC_SIZES]
    
    n_neighbors = min(3, len(minority_class))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(minority_class)
    for synthetic_size in SYNTHETIC_SIZES:
        synthetic_data = []
        for _ in range(synthetic_size):
            sample = minority_class[np.random.randint(0, len(minority_class))]
            _, indices = neighbors.kneighbors([sample])
            kneighbors = minority_class[indices][0]

            neighbor = kneighbors[np.random.randint(0, len(kneighbors))]
            fraction = random.random()

            synthetic_point = sample + (neighbor - sample) * fraction
            synthetic_data.append(synthetic_point)
        synthetic_data = pd.DataFrame.from_records(synthetic_data, columns=["ID_IG", "Time", "Temperature", "BET", "Selectivity"])
        results.append(synthetic_data)
    return results


def add_keys(fold_dict: dict, results: dict) -> None:
    for key, value in fold_dict.items():
        if key not in results:
            results[key] = ([], [], [])
        for i in range(len(value)):
            results[key][i].append(value[i])


def pipeline(models: dict, baseline: pd.DataFrame, input_data: list, output_data: str) -> dict:
    results = dict()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for (train, test) in kf.split(baseline[input_data], baseline[output_data]):
        data = {"training_set": baseline.iloc[train]}
        metadata = Metadata.detect_from_dataframes(data)
        baseline_fold = baseline_evaluation(models, baseline, train, test)
        add_keys(baseline_fold, results)

        for name, synthesizer in create_synthesizers(metadata).items():
            if name == "smote":
                synthetic = smote_synthetic_generation(baseline, train)
            else:
                synthesizer.fit(data["training_set"])
                synthetic = [synthesizer.sample(num_rows=n) for n in SYNTHETIC_SIZES]
            datasets = data_division(baseline, synthetic)

            for model in models:
                synthetic_fold = synthetic_evaluation(models, datasets, baseline, name, test, model)
                add_keys(synthetic_fold, results)         
    results = {
        key: (round(mean(value[0]), 3), round(mean(value[1]), 3), round(mean(value[2]), 3))
        for key, value in results.items()}
    return sorted(results.items(), key=lambda score: score[1][0], reverse=True)[0:5]
        

if __name__ == "__main__":
    main()