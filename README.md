# RNN-LSTM-based HAR Model Design

Design and training of a stacked-LSTM (RNN) model for **Human Activity Recognition (HAR)** from 3-axis accelerometer signals, with export of the trained model to TensorFlow Lite for on-device inference.

The notebooks were written for **Google Colab** with the project stored on Google Drive (`/content/drive/My Drive/TA/Perancangan model Human Activity Recognition/`).

## Stack

Python, TensorFlow / Keras, scikit-learn, NumPy, pandas, SciPy, Matplotlib / seaborn / Plotly, Optuna (imported), Kaggle API.

## Notebooks

| Notebook | What it does |
| --- | --- |
| `download_dataset.ipynb` | Mounts Google Drive and downloads the Kaggle dataset `erenaktas/human-activity-recognition` with the Kaggle CLI, then unzips it. |
| `prepare_dataset.ipynb` | Downloads the Kaggle dataset `malekzadeh/motionsense-dataset` and plots a sample confusion-matrix heatmap (sitting / jumping / walking / running) with Plotly. |
| `acc-rnn-lstm-har-rev10.ipynb` | 4-class HAR (`sitting`, `jumping`, `walking`, `running`). Extracts 3 accelerometer columns (columns 18–20, first 125 rows) from each `ACC_Dataset` segment into `HAR_ACC_Dataset`, windows the data (window 100, overlap 75), trains with leave-subject-out validation permutations on subjects p1–p8 (subjects 6 and 8 held out for testing), plots a confusion matrix and converts the chosen model to TFLite. Saves models to `plusjump_model/` (not included in this repo). |
| `acc-rnn-lstm-har-rev11.ipynb` | 2-class variant (`Resting` vs `No Resting`). Converts the raw CSV recordings in `LAR_dataset` (x, y, z accelerometer columns, 2600 rows per file) into `LAR_ACC_Dataset`, windows the data (window 100, overlap 80), trains with 10-fold `KFold`, saves models to `our_model/`, exports a SavedModel and converts `our_model/2_64/model_w0100_o080_e50_t10000.h5` to TFLite. |

### Model

Defined in `create_model()` in the rev10/rev11 notebooks:

```
Input (window, 3)
-> Dense(dense1)               (optional)
-> BatchNormalization
-> LSTM(lstm1) [-> Dropout -> LSTM(lstm2)] [-> Dropout -> LSTM(lstm3)]
-> Dropout(0.2)
-> Dense(n_classes, softmax)
```

Loss `categorical_crossentropy`, optimizer `adam`, up to 50 epochs with `EarlyStopping` on `val_accuracy` (patience 10). Before windowing, `clean_data()` drops all-zero rows, interpolates NaNs and removes isolated single-sample spikes.

## Repository structure

```
ACC_Dataset/<activity>/p<1-8>/sNN.txt     # raw segments, 125 rows x 45 comma-separated columns
                                           # activities: downstairs, jumping, running, sitting,
                                           # standing, upstairs, walking, walking_v1
HAR_ACC_Dataset/p<1-8>/<activity>/sNN.txt  # 3-axis accelerometer extracted by rev10 (4 activities, 60 segments each)
LAR_dataset/p1/<Resting|No Resting>/data (N).csv   # raw phone recordings (15 per class)
LAR_ACC_Dataset/p1/<Resting|No Resting>/sNN.txt    # 3-axis accelerometer extracted by rev11
our_model/
  <L>_<U>/model_w<window>_o<overlap>_e<epochs>_t<acc>.h5
  saved_models/LSTM_model/                 # TensorFlow SavedModel (batch 1, 100 steps, 3 inputs)
  tensorflowlite_model/model_HAR_RNNLSTM_ACC_new.tflite
```

### Model file naming (`our_model/`)

- Folder `<L>_<U>`: number of LSTM layers and units per layer, e.g. `3_64` = three LSTM layers of 64 units.
- `w0100`: window length (100 samples).
- `o080`: window overlap (80 samples).
- `e50`: maximum training epochs.
- `t<acc>`: accuracy on the evaluation fold multiplied by 10,000 and rounded (e.g. `t9895` = 98.95%, `t10000` = 100%).

## How to run

1. Open a notebook in Google Colab and mount Google Drive.
2. Place this repository's folders in your Drive and adjust `working_dir` at the top of the notebook if your path differs.
3. To download datasets with the Kaggle CLI, put **your own** `kaggle.json` API token in the folder pointed to by `KAGGLE_CONFIG_DIR`. It is ignored by git and must never be committed.
4. Run the cells in order: dataset extraction, `create_dataset(...)`, training (`train_session()`), evaluation and TFLite conversion.
