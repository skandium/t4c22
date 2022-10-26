import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import argparse

from utils import load_preprocessed_counters

from conf import data_dir

NUM_PCS = 10

def calculate_numeric_volumes(counters):
    eng_vals = []
    for v in tqdm(counters["volumes_1h"]):
        if np.isnan(v).any():
            # If entire array is none, fall back to 0
            if np.isnan(v).sum() == 4:
                arr = np.nan_to_num(v)
            else:
                # Otherwise, try median
                arr = np.nan_to_num(v, nan=np.nanmedian(v))
        else:
            arr = v
        eng_vals.append(np.sum(arr))
    return eng_vals



# Andrei's magic PCA
class PCATrafficFeatures:
    def __init__(self, num_components=20, feature_base_name="PCA",
                 count_agg_output_feature="agg_volumes"):
        self.standard_scaler = StandardScaler()
        self.PCA = PCA(n_components=num_components)
        self.feature_base_name = feature_base_name
        self.num_components = num_components
        self.count_agg_output_feature = count_agg_output_feature
        tqdm.pandas()

    def get_pivot_counter_time(self, counts, grouper=["day", "t"]):
        counts_fallbacks = counts.groupby("node_id")[self.count_agg_output_feature].median().to_dict()
        counts[self.count_agg_output_feature] = [
            counts_fallbacks[node_id] if pd.isna(median) else median
            for median, node_id in zip(counts[self.count_agg_output_feature], counts["node_id"])]
        counts[self.count_agg_output_feature] = counts[self.count_agg_output_feature].fillna(
            counts[self.count_agg_output_feature].median())

        df_pivot_counter_time = counts.pivot(index=grouper, columns="node_id", values=self.count_agg_output_feature)
        counter_medians = df_pivot_counter_time.median()
        df_pivot_counter_time = df_pivot_counter_time.fillna(counter_medians)

        return df_pivot_counter_time

    def get_pca_features(self, df_pivot_counter_time, mode="train"):
        assert mode in {"train", "test"}

        if mode == "train":
            self.standard_scaler.fit(df_pivot_counter_time)
        counts_pca_scaled = self.standard_scaler.transform(df_pivot_counter_time)

        if mode == "train":
            self.PCA.fit(counts_pca_scaled)
        pca_features = self.PCA.transform(counts_pca_scaled)

        pca_reduced = pd.DataFrame(index=df_pivot_counter_time.index, data=pca_features)
        pca_reduced = pca_reduced.reset_index()

        return pca_reduced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", "-c", required=True)
    parser.add_argument("--feature", "-f", required=True)
    args = parser.parse_args()
    city_name = args.city_name
    feature = args.feature

    traffic_path = data_dir / "traffic" / city_name
    traffic_path.mkdir(parents=True, exist_ok=True)

    counters_train = load_preprocessed_counters(city_name, "train")
    counters_train["test_idx"] = -1

    print(counters_train.shape)
    counters_test = load_preprocessed_counters(city_name, "test")
    counters_test["day"] = "2050-01-01"
    counters_test["t"] = counters_test["test_idx"]
    print(counters_test.shape)
    counts = pd.concat([counters_train, counters_test])
    print(counts.shape)

    PCATF = PCATrafficFeatures(
        num_components=NUM_PCS,
        feature_base_name="PCA",
        count_agg_output_feature=feature
    )
    print("Pivoting train")
    df_pivot_train = PCATF.get_pivot_counter_time(counts)
    del counts
    print("Fetching train PCAs")
    pca_features_train = PCATF.get_pca_features(df_pivot_train, mode="train")
    pca_features_train.columns = ["day", "t"] + [f"PC_{i}" for i in range(NUM_PCS)]
    print("Saving")
    pca_features_train.to_parquet(traffic_path / f"pcs_series_train_{feature}.parquet")

    print("Dealing with test")
    counts_test = load_preprocessed_counters(city_name, "test")
    df_pivot_test = PCATF.get_pivot_counter_time(counts_test, ["test_idx"])
    df_pivot_test_backfill = pd.DataFrame(columns=df_pivot_train.columns)
    for x in df_pivot_train.columns:
        try:
            df_pivot_test_backfill[x] = df_pivot_test[x]
        except Exception as e:
            print(f"Missing column: {x}")
            df_pivot_test_backfill[x] = [0] * 100
    pca_features_test = PCATF.get_pca_features(df_pivot_test_backfill, mode="test")
    pca_features_test.columns = ["test_idx"] + [f"PC_{i}" for i in range(NUM_PCS)]
    pca_features_test.to_parquet(traffic_path / f"pcs_series_test_{feature}.parquet")
