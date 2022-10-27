#!/usr/bin/env bash

set -Eeuox pipefail

python precompute_volume_features.py -c $1 #(Comment out median if you're in a hurry)
python generate_principal_components.py -c $1 -f volumes_last
python generate_principal_components.py -c $1 -f volumes_sum