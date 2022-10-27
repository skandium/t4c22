# IARAI Traffic4cast 2022

This should roughly replicate the solution of the Bolt team. Code is (slightly) refactored compared to the messy notebooks
that were used during competition.

## Generic preprocessing (needs to be run only once)

```
pip install -r requirements.txt
bash preprocess_all.sh melbourne
```

## Core

Run the notebook `core_create_target_encodings.ipynb`
Now you're ready to run the notebook `core_final.ipynb`! Note that it raises an intentional error after training and
before creating submissions, but can be resumed manually after choosing the desired number of iterations to use.

## Extended


## Artifacts
We store the trained model artifacts together with submission parquet files in S3.
