# IARAI Traffic4cast 2022

This should roughly replicate the solution of the Bolt team. Code is (slightly) refactored compared to the messy
notebooks that were used during competition.

Python version: any new-ish Python 3.8+ should work, we used 3.10.4

Change `data_dir` in `conf.py` to where all the competition data (`train`, `test`, `road_graph` etc) is.

## Generic preprocessing (needs to be run only once) for both tracks

```
pip install -r requirements.txt
bash preprocess_all.sh melbourne
```

An alternative to running this and task specific preprocessing is unarchiving the `traffic.zip` file and moving  resulting `traffic` directory
to `data_dir`.

## Core

Preprocessing:
Run the notebook `core_create_target_encodings.ipynb`

Now you're ready to run the notebook `core_final.ipynb`! Note that it raises an intentional error after training and
before creating submissions, but can be resumed manually after choosing the desired number of iterations to use.

### Making submissions

To only generate a submission from a trained model, run:

```
python core_generate_submission.py -c melbourne -p model_path -m model_name
```

This will create a submission in `data_dir` (more
specifically, `data_dir / "submissions" / model_name / city_name / "labels" / "cc_labels_test.parquet"`)

## Extended

Preprocessing:
Run the notebook `extended_generate_supersegment_speed_feats.ipynb`
Now run the notebook `extended_final.ipynb`.

### Making submissions

To only generate a submission from a trained model, run:

```
python extended_generate_submission.py -c melbourne -p model_path -m model_name
```

## Artifacts

We store the trained model artifacts together with submission parquet files in S3:

- preprocessed `traffic` [folder](https://t4c22-data.s3.eu-central-1.amazonaws.com/traffic.zip)
- core [submissions](https://t4c22-data.s3.eu-central-1.amazonaws.com/submissions/core.zip)
- core [models](https://t4c22-data.s3.eu-central-1.amazonaws.com/models/core.zip)
- extended [submissions](https://t4c22-data.s3.eu-central-1.amazonaws.com/submissions/extended.zip)
- extended [models](https://t4c22-data.s3.eu-central-1.amazonaws.com/models/extended.zip)
