# How to Run Deep-Retina Models

## 📋 Parameters Explanation

### `--expt` (Experiment Date)
The date of the experiment in `YY-MM-DD` format. Available options:
- `15-10-07`
- `15-11-21a`
- `15-11-21b`
- `16-01-07`
- `16-01-08`
- `16-05-31`

### `--stim` (Stimulus Type)
The type of visual stimulus used:
- `whitenoise` - White noise stimulus
- `naturalscene` - Natural scene images

### `--model` (Model Architecture)
The neural network architecture to train:
- `BN_CNN` - Batch Normalized Convolutional Neural Network
- `LN_softplus` - Linear-Nonlinear model with softplus activation
- `LN_sigmoid` - Linear-Nonlinear model with sigmoid activation
- `LN_relu` - Linear-Nonlinear model with ReLU activation
- `LN_rbf` - Linear-Nonlinear model with RBF (Radial Basis Function) activation

### `--cell` (Cell Index, only for LN models)
The index of the retinal ganglion cell to train on. Required only for LN models.
- For `15-10-07`: cells `0, 1, 2, 3, 4`
- For `15-11-21a`: cells `6, 10, 12, 13`
- For `15-11-21b`: cells `0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25`
- For `16-01-07`: cells `0, 2, 7, 10, 11, 12, 31`
- For `16-01-08`: cells `0, 3, 7, 9, 11`
- For `16-05-31`: cells `2, 3, 4, 14, 16, 18, 20, 25, 27`

## 🚀 Example Commands

### Train BN_CNN model:
```bash
cd deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN
```

### Train LN model (requires --cell):
```bash
cd deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model LN_softplus --cell 0
```

## ⚠️ Important: Data Requirements

**The code expects experimental data files at:**
```
~/experiments/data/<expt>/<stim>.h5
```

For example:
```
~/experiments/data/15-10-07/whitenoise.h5
~/experiments/data/15-10-07/naturalscene.h5
```

**These data files are NOT included in the repository.** You need to:
1. Obtain the experimental data files (HDF5 format)
2. Place them in the correct directory structure: `~/experiments/data/<expt>/<stim>.h5`
3. The HDF5 files should contain:
   - `train/stimulus` - Stimulus data
   - `train/response/firing_rate_10ms` - Firing rate responses
   - `train/response/binned` - Binned spike counts
   - `train/stas/cell<NN>` - Spike-triggered averages (for cutout)

## 📝 Data File Structure

Each HDF5 file should have this structure:
```
<stim>.h5
├── train/
│   ├── stimulus          # [time, height, width] array
│   ├── time              # Time vector
│   ├── response/
│   │   ├── firing_rate_10ms  # [ncells, time] array
│   │   └── binned            # [ncells, time] array
│   └── stas/
│       └── cell<NN>      # Spike-triggered average for each cell
└── test/
    └── (same structure as train/)
```

## 🔧 If You Don't Have Data

If you don't have the experimental data files, you cannot run the training scripts directly. However, you can:

1. **Check the model definitions** - Look at `deepretina/models.py` to understand the architectures
2. **Use with your own data** - Modify `deepretina/experiments.py` to load your data format
3. **Contact the authors** - The original paper may have data availability information

## 📚 References

- Paper: [Deep Learning Models of the Retinal Response to Natural Scenes](https://arxiv.org/abs/1702.01825)
- Repository: https://github.com/baccuslab/deep-retina

