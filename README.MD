# IARAI Traffic4cast 2022

This should roughly replicate the solution of the Bolt team. Code is (slightly) refactored compared to the messy notebooks
that were used during competition.

## Generic preprocessing (needs to be run only once)

* `python precompute_volume_features.py -c melbourne`
* `python generate_principal_components.py -c melbourne -f volumes_last`
* `python generate_principal_components.py -c melbourne -f volumes_sum`

## Core

Run the notebook `core_create_target_encodings.ipynb`
Now you're ready to run the notebook `core_final.ipynb`!

## Extended

