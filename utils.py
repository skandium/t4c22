import numpy as np
from tqdm import tqdm
import pandas as pd
import hdmedians as hd
from conf import data_dir


def calculate_volume_features(counters, aggregators: dict, nan_to_zero=False):
    print("Calculating volume features")

    for f in aggregators:
        print(f)
        eng_vals = []
        for v in tqdm(counters["volumes_1h"]):
            if np.isnan(v).any():
                if np.isnan(v).sum() == 4:
                    if nan_to_zero:
                        # If entire array is none, fall back to 0
                        arr = np.nan_to_num(v)
                    else:
                        arr = v
                else:
                    # Otherwise, try median
                    arr = np.nan_to_num(v, nan=np.nanmedian(v))
            else:
                arr = v
            eng_vals.append(aggregators[f](arr))
        counters[f] = eng_vals
    return counters


def load_counters(city_name, mode):
    print(city_name)
    count_frames = []
    for file in tqdm(sorted((data_dir / mode / city_name / 'input').glob('counters_*.parquet'))):
        count_frames.append(pd.read_parquet(file))
    print(f'Read {len(count_frames)} training input files for {city_name}')
    counts = pd.concat(count_frames)
    del count_frames
    print(counts.shape)
    return counts


def load_preprocessed_counters(city_name, mode):
    counts = pd.read_parquet(data_dir / "traffic" / city_name / f"all_counters_{mode}.parquet")
    print(counts.shape)
    return counts


def load_labels_core(city_name, edge_id_to_int: dict, sample=None):
    train_input_frames = []
    label_files = sorted((data_dir / 'train' / city_name / 'labels').glob('cc_labels_*.parquet'))

    for train_input_file in tqdm(label_files):
        train_input_frames.append(pd.read_parquet(train_input_file))
    labels = pd.concat(train_input_frames)
    del train_input_frames
    print(f"loaded {len(labels)} label rows")

    if sample:
        print(f"sampling {int(sample)} rows")
        labels = labels.sample(int(sample), random_state=42)

    labels["edge_id"] = [f"{u}_{v}" for u, v in zip(labels["u"], labels["v"])]
    labels["edge_int"] = [edge_id_to_int[eid] for eid in labels["edge_id"]]

    # Save some memory, maybe
    # del labels["v"]
    del labels["edge_id"]
    return labels


def get_medoid(lst):
    return hd.medoid(np.array(lst).T)


def blacklist_counters(city_name, threshold=0.8):
    # # Blacklist inactive counters - we won't carry them in our joins nor use for neighbour features
    counters_train = load_counters(city_name, "train")
    counters_train["test_idx"] = -1
    counters_test = load_counters(city_name, "test")
    counters_test["day"] = -1
    counters_test["t"] = -1
    counters = pd.concat([counters_train, counters_test])
    print(counters["node_id"].nunique())

    counter_threshold = counters["node_id"].value_counts().max() * threshold
    good_counters = counters["node_id"].value_counts()[counters["node_id"].value_counts() > counter_threshold].index

    counters = counters[counters["node_id"].isin(good_counters)]
    print(counters["node_id"].nunique())
    del counters["volumes_1h"]
    del counters["day"]
    del counters["t"]
    del counters["test_idx"]
    counters = counters.drop_duplicates()
    print(counters.shape)
    return counters


def create_nodes_with_counters(city_name, blacklist=False):
    nodes = pd.read_parquet(data_dir / f"road_graph/{city_name}/road_graph_nodes.parquet")

    if blacklist:
        counters = blacklist_counters(city_name)
    else:
        sample_counters = {
            "melbourne": "2020-06-01",
            "london": "2019-07-01",
            "madrid": "2021-06-01"
        }

        # Use one counter slice to find nearest counters
        # TODO - annoyingly, the counters seem to be a (slightly) changing set, need to handle this
        # TODO calculate superset of all counters, looping over files
        counters = pd.read_parquet(data_dir / f"train/{city_name}/input/counters_{sample_counters[city_name]}.parquet")
        counters = counters[counters.t == 4]
        del counters["volumes_1h"]

    # First, create counter_ids
    nodes_with_counters = pd.merge(nodes, counters, on="node_id")
    nodes_with_counters["counter_id"] = list(range(len(nodes_with_counters)))
    print(nodes_with_counters["counter_id"].nunique())
    return nodes_with_counters


def merge_pcas(city_name, data, mode="train"):
    print(data.shape)
    # Merge city traffic PCs
    print("Merging traffic PCs")
    grouper = ["day", "t"] if mode == "train" else ["test_idx"]
    df_pcas = pd.read_parquet(data_dir / "traffic" / city_name / f"pcs_series_{mode}_volumes_last.parquet",
                              columns=grouper + [f"PC_{i}" for i in range(8)])
    data = data.merge(df_pcas, on=grouper, suffixes=("", f"_volumes_last"), how="left")
    print(data.shape)

    for volume_feat in ["volumes_sum"]:
        # Merge city traffic PCs
        print("Merging traffic PCs")
        df_pcas = pd.read_parquet(data_dir / "traffic" / city_name / f"pcs_series_{mode}_{volume_feat}.parquet",
                                  columns=grouper + [f"PC_{i}" for i in range(5)])
        data = data.merge(df_pcas, on=grouper, suffixes=("", f"_{volume_feat}"), how="left")
        print(data.shape)

    print(data.columns)
    return data


def split_train_valid(city_name, data):
    valid_weeks_hardcoded = {
        # From [23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53]
        "melbourne": [25, 33, 41, 49],
        # From [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52]
        "madrid": [24, 32, 40, 48],
        # From [27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51,  1,  3,  5]
        "london": [29, 37, 45, 1]
    }

    if city_name == "madrid":
        # Hack the timestamps so weeks are contained
        data["date"] = pd.to_datetime(data["day"]) - pd.Timedelta(days=1)
        data["week"] = data["date"].dt.week
    else:
        data["date"] = pd.to_datetime(data["day"])
        data["week"] = data["date"].dt.week

    unique_weeks = data["week"].unique()
    print("All weeks")
    print(sorted(unique_weeks))

    if city_name not in valid_weeks_hardcoded:
        raise ValueError("Define forced spread!")

    valid_weeks = valid_weeks_hardcoded[city_name]

    print("Valid weeks")
    print(valid_weeks)

    train = data[~data["week"].isin(valid_weeks)]
    valid = data[data["week"].isin(valid_weeks)]

    del train["week"]
    del train["date"]
    del valid["week"]
    del valid["date"]

    print(f"Train shape: {train.shape}")
    print(f"Valid shape: {valid.shape}")

    return train, valid


def load_edges(city_name):
    edges = pd.read_parquet(data_dir / f"road_graph/{city_name}/road_graph_edges.parquet")
    edges["edge_id"] = [f"{u}_{v}" for u, v in zip(edges["u"], edges["v"])]
    edge_id_to_int = {k: v for k, v in zip(edges["edge_id"], list(range(len(edges))))}
    edge_int_to_id = {v: k for k, v in zip(edges["edge_id"], list(range(len(edges))))}
    edges["edge_int"] = [edge_id_to_int[eid] for eid in edges["edge_id"]]
    return edges, edge_id_to_int, edge_int_to_id


def proba_to_logit(p):
    return np.log(p + 0.0000000000000000001)


def get_weights_from_class_fractions(class_fractions):
    n = np.sum(class_fractions)
    return [n / (c * 3) for c in class_fractions]

def load_supersegments(city_name, node_coordinates, del_segment_feats=True):
    supersegments = pd.read_parquet(data_dir / f"road_graph/{city_name}/road_graph_supersegments.parquet")
    supersegments["supersegment_id"] = list(range(len(supersegments)))
    supersegment_to_id = supersegments.groupby("identifier")["supersegment_id"].first().to_dict()
    id_to_supersegment = supersegments.groupby("supersegment_id")["identifier"].first().to_dict()

    # Get representative point of supersegment
    tqdm.pandas()

    supersegments_full = supersegments.explode("nodes")
    supersegments_full["coords"] = [(node_coordinates[n]["x"], node_coordinates[n]["y"]) for n in
                                    supersegments_full["nodes"]]
    supersegments["coord_list"] = \
        supersegments_full.groupby("supersegment_id")["coords"].progress_apply(lambda x: x.tolist()).reset_index()[
            "coords"]
    del supersegments_full
    supersegments["representative_point"] = supersegments["coord_list"].progress_apply(get_medoid)
    supersegments["x"] = [r[0] for r in supersegments["representative_point"]]
    supersegments["y"] = [r[1] for r in supersegments["representative_point"]]
    if del_segment_feats:
        del supersegments["nodes"]
        del supersegments["coord_list"]
        del supersegments["representative_point"]
        del supersegments["identifier"]
    return supersegments, supersegment_to_id, id_to_supersegment


def create_multi_traffic_state(data, traffic_quantiles_list, feature="volumes_mean", mode="train"):
    if mode == "train":
        grouper = ["day", "t"]
    else:
        grouper = ["test_idx"]

    traffic_means = data.groupby(grouper)[feature].median().reset_index()
    features_quantiles = []

    for i, traffic_quantiles in enumerate(traffic_quantiles_list):
        curr_quantile_feature = f"quantile_{i}"
        features_quantiles.append(curr_quantile_feature)

        traffic_means[curr_quantile_feature] = [(vol - traffic_quantiles).abs().argmin() for vol in
                                                traffic_means[feature]]

    traffic_means["total_traffic"] = traffic_means[feature]
    del traffic_means[feature]
    data = data.merge(traffic_means, on=grouper)

    return data, features_quantiles


def append_static_pred_relative_to_df(df_target, df_reference, input_quantile_field="quantile",
                                      output_static_pred_field="static_pred"):
    """ Append ETA static_preds to df_target based on df_reference quantile valus

    :param df_target: df that will get the static_pred ETAs appended
    :param df_reference: df that will be used for computing the quantile-based supersegment ETAs
    :returns: doesn't return anything, it just changes its first argument
    """

    eta_dummy_fallback = df_reference["eta"].median()
    eta_dict_fallback = df_reference.groupby(['supersegment_id'])["eta"].median().to_dict()
    eta_dict = df_reference.groupby(['supersegment_id', input_quantile_field])["eta"].median().to_dict()
    eta_dict_counts = df_reference.groupby(['supersegment_id', input_quantile_field])["eta"].count().to_dict()

    df_target[output_static_pred_field] = [
        eta_dict.get((s, q), eta_dict_fallback.get(s, eta_dummy_fallback))
        for s, q in zip(
            df_target["supersegment_id"],
            df_target[input_quantile_field]
        )
    ]