import random
import re
import argparse
import time
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.neighbors import KDTree, BallTree
from pathlib import Path

from utils import create_nodes_with_counters, merge_pcas, load_edges
from conf import data_dir


def create_categorical_features(data):
    feats = ["oneway", "highway", "tunnel"]

    feature_dicts = {}
    # Encode categorical features
    for f in feats:
        categories = data[f].astype("category")
        cat_codes = categories.cat.codes
        data[f"{f}_cat"] = cat_codes
        feature_dicts[f] = {k: v for k, v in zip(data[f], cat_codes)}

    return data, feature_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", "-c", required=True)
    parser.add_argument("--model_path", "-m", required=True)
    args = parser.parse_args()
    city_name = args.city_name
    model_path = args.feature

    gbm = lgb.Booster(model_file=model_path)

    nodes = pd.read_parquet(data_dir / f"road_graph/{city_name}/road_graph_nodes.parquet")

    edges, edge_id_to_int, edge_int_to_id = load_edges(city_name)
    edges, edge_dicts = create_categorical_features(edges)

    # Get representative point of edge
    edges = edges.merge(nodes[["node_id", "x", "y"]], left_on="u", right_on="node_id", how="left")

    nodes_with_counters = create_nodes_with_counters(city_name, blacklist=False)
    # Find nearest counter for edges which are not immediately at counter
    tree = KDTree(nodes_with_counters[["x", "y"]], metric="euclidean")
    dist, ind = tree.query(edges[["x", "y"]], k=NEIGHBORS_FOR_WEIGHTING)
    edges["nearest_counter_id"] = ind[:, 0]
    edges["counter_distance_euclidean"] = dist[:, 0]
    edges["counter_distance_euclidean_mean_all"] = dist.mean(axis=1)

    test_path = data_dir / "test" / city_name / "input" / "counters_test.parquet"
    counters_test = pd.read_parquet(test_path)

    pd.options.mode.chained_assignment = None  # default='warn'

    # For test set, we need to create a submission set of length len(edges) * counters_test["test_idx"].nunique()
    # Do this in iterations, as direct join returned weird DF shape
    full_test = []
    for t in tqdm(range(counters_test["test_idx"].nunique())):
        full = edges.copy()
        full["test_idx"] = t
        full_test.append(full)

    full_test = pd.concat(full_test)

    volume_agg_test = pd.read_parquet(traffic_path / city_name / f"volume_agg_test.parquet")
    full_test = full_test.merge(volume_agg_test, on=["test_idx"])
    full_test = merge_pcas(city_name, full_test, mode="test")

    cc_distributions = pd.read_parquet(data_dir / "traffic" / city_name / "cc_dist.parquet")
    full_test = full_test.merge(cc_distributions, on="edge_int")

    bomber_feats = pd.read_parquet(data_dir / "traffic" / city_name / "bomber_feats.parquet")
    full_test = full_test.merge(bomber_feats, on="edge_int")

    full_test["proba_green"] = np.exp(full_test["logit_green"]) / (
            np.exp(full_test["logit_green"]) + np.exp(full_test["logit_yellow"]) + np.exp(full_test["logit_red"]))
    full_test["proba_yellow"] = np.exp(full_test["logit_yellow"]) / (
            np.exp(full_test["logit_green"]) + np.exp(full_test["logit_yellow"]) + np.exp(full_test["logit_red"]))
    full_test["proba_red"] = np.exp(full_test["logit_red"]) / (
            np.exp(full_test["logit_green"]) + np.exp(full_test["logit_yellow"]) + np.exp(full_test["logit_red"]))

    features = gbm.feature_name()
    label = "cc"

    for f in features:
        assert f in full_test.columns, f

    stm = time.time()
    gbm_preds = gbm.predict(full_test[features], raw_score=True)
    print(f"Took {time.time() - stm} seconds")

    full_preds = gbm_preds + full_test[["logit_green", "logit_yellow", "logit_red"]]

    full_test["logit_green"] = full_preds["logit_green"].round(3)
    full_test["logit_yellow"] = full_preds["logit_yellow"].round(3)
    full_test["logit_red"] = full_preds["logit_red"].round(3)

    assert full_test["test_idx"].nunique() * full_test["edge_int"].nunique() == len(full_test)

    print(full_test[["logit_green", "logit_yellow", "logit_red"]].quantile(
        q=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))

    submission_path = data_dir / "submissions" / model_name / city_name / "labels" / "cc_labels_test.parquet"

    submission_features = [
        "logit_green",
        "logit_yellow",
        "logit_red",
        "u",
        "v",
        "test_idx"
    ]

    import time

    stm = time.time()
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    full_test[submission_features].to_parquet(submission_path)
    print(f"Took {time.time() - stm} seconds")
