from utils import load_counters, calculate_volume_features
import argparse
import numpy as np
from conf import data_dir



# Feature engineering on top of counter series
def gr(vals):
    return (vals[-1] / (vals[0] + 0.01))


def argmax(vals):
    return max(enumerate(vals), key=lambda x: x[1])[0]


def amplitude(vals):
    min_vals = min(vals)
    if min_vals == 0:
        return -1
    else:
        return max(vals) / min(vals)


def last(vals):
    return vals[-1]


engineered_volume_features = {
    # "volumes_amplitude": amplitude,
    "volumes_gr": gr,
    # "volumes_argmax": argmax,
    # "volumes_max": np.max,
    # "volumes_min": np.min,
    # "volumes_mdn": np.median,
    # "volumes_mean": np.mean,
    "volumes_sum": np.sum,
    "volumes_last": last
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", "-c", required=True)
    args = parser.parse_args()
    city_name = args.city_name

    counters = load_counters(city_name, "train")
    counters = calculate_volume_features(counters, aggregators=engineered_volume_features, nan_to_zero=True)
    del counters["volumes_1h"]
    counters.to_parquet(data_dir / "train" / city_name / "input" / f"all_counters.parquet")

    counters = load_counters(city_name, "test")
    counters = calculate_volume_features(counters, aggregators=engineered_volume_features, nan_to_zero=True)
    del counters["volumes_1h"]
    counters.to_parquet(data_dir / "test" / city_name / "input" / f"all_counters.parquet")
