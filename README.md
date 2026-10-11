# IRIS ML Pipeline on Vertex AI — Week 1

A minimal pipeline for the IRIS classifier: data in GCS → training on Vertex AI Workbench → artifacts saved to GCS in folders named by timestamp → a separate inference script.

```
data/v1/iris.csv, data/v2/iris.csv    raw dataset versions
src/prepare_data.py                   split into train/eval, upload to gs://BUCKET/data/<version>/
src/train.py                          train, save to gs://BUCKET/artifacts/<version>/<timestamp>/
src/inference.py                      load the model from GCS, predict on eval.csv
src/compare_runs.py                   compare all runs and data versions
run_pipeline.sh                       runs train + inference twice per data version
```

GCS layout after a full run:

```
gs://BUCKET/
├── data/v1/{train.csv, eval.csv}
├── data/v2/{train.csv, eval.csv}
└── artifacts/
    ├── v1/2026-10-12T10-30-01/{model.joblib, metrics.json, train.log, predictions.csv, inference_metrics.json}
    ├── v1/2026-10-12T10-30-04/...
    ├── v2/...
```

## Task 1 — GCP and Vertex AI Workbench

1. Activate the free trial at https://console.cloud.google.com and create a project (e.g. `mlops-week1`).
2. Enable these APIs (**APIs & Services → Library**): **Vertex AI API**, **Notebooks API**, **Compute Engine API**, **Cloud Storage API**.
   Or from Cloud Shell:
   ```bash
   gcloud services enable aiplatform.googleapis.com notebooks.googleapis.com compute.googleapis.com storage.googleapis.com
   ```
3. **Vertex AI → Workbench → Instances → Create new**. Pick a region such as `us-central1` and the default machine type (e4/n1-standard-4 is enough). Click **Open JupyterLab**.

## Task 2 — Upload data to GCS

In a Workbench terminal:

```bash
git clone -b week_1 https://github.com/<you>/<ROLL_NO>_MLOPS_WEEKLY_ASSIGNMENT.git
cd <ROLL_NO>_MLOPS_WEEKLY_ASSIGNMENT
pip install -r requirements.txt

export BUCKET=gs://<roll-no>-mlops-iris          # must be globally unique
gcloud storage buckets create $BUCKET --location=us-central1
python src/prepare_data.py --bucket $BUCKET --version v1
gcloud storage ls -r $BUCKET/data
```

The Workbench service account can already read and write buckets in the same project, so you don't need any keys.

## Tasks 3 & 4 — Train, then run inference

```bash
python src/train.py --bucket $BUCKET --version v1        # prints the run id (timestamp)
python src/inference.py --bucket $BUCKET --version v1    # uses the latest run by default
```

## Task 5 — Run twice

```bash
./run_pipeline.sh $BUCKET v1
gcloud storage ls $BUCKET/artifacts/v1/                  # two timestamp folders
```

## Task 6 (optional) — Two data versions

```bash
./run_pipeline.sh $BUCKET v1 v2
```

`compare_runs.py` prints validation and eval accuracy for every run of each version. If the course repo's `data` folder has its own versions, copy them over `data/v1/iris.csv` and `data/v2/iris.csv` first. The files here are standard IRIS (v1) and IRIS plus 150 noisy augmented rows (v2).

## Local testing (no GCP)

Every script also accepts a local folder in place of `gs://`:

```bash
./run_pipeline.sh /tmp/fakebucket v1 v2
```
