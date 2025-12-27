# Example Commands for Deep-Retina

## 📋 What to Fill In

### Full Example:

```bash
cd deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN
```

## 🔤 Parameter Explanation

### `--expt` (Experiment Date)
**Available values:**
- `15-10-07`
- `15-11-21a`
- `15-11-21b`
- `16-01-07`
- `16-01-08`
- `16-05-31`

**Example:** `--expt 15-10-07`

---

### `--stim` (Stimulus Type)
**Available values:**
- `whitenoise` - White noise stimulus
- `naturalscene` - Natural scene images

**Example:** `--stim whitenoise`

---

### `--model` (Model Architecture)
**Available values:**

**For CNN model:**
- `BN_CNN` - Batch Normalized Convolutional Neural Network

**For Linear-Nonlinear (LN) models:**
- `LN_softplus` - with softplus activation
- `LN_sigmoid` - with sigmoid activation
- `LN_relu` - with ReLU activation
- `LN_rbf` - with Radial Basis Functions

**Example:** `--model BN_CNN`

---

### `--cell` (Cell Index)
**Required only for LN models**

**Available values (depends on experiment):**
- For experiment `15-10-07`: `0, 1, 2, 3, 4`
- For experiment `15-11-21a`: `6, 10, 12, 13`
- For experiment `15-11-21b`: `0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25`
- For experiment `16-01-07`: `0, 2, 7, 10, 11, 12, 31`
- For experiment `16-01-08`: `0, 3, 7, 9, 11`
- For experiment `16-05-31`: `2, 3, 4, 14, 16, 18, 20, 25, 27`

**Example:** `--cell 0`

---

## 📝 Complete Command Examples

### 1. Run BN_CNN model:
```bash
cd deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model BN_CNN
```

### 2. Run LN (Linear-Nonlinear) model:
```bash
cd deep-retina-master/scripts
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim whitenoise --model LN_softplus --cell 0
```

### 3. More examples:
```bash
# With natural scenes
../../.conda/bin/python fit_models.py --expt 15-10-07 --stim naturalscene --model BN_CNN

# With different experiment
../../.conda/bin/python fit_models.py --expt 15-11-21a --stim whitenoise --model LN_sigmoid --cell 6
```

---

## ⚠️ Important Note

**Code expects data at:**
```
~/experiments/data/<expt>/<stim>.h5
```

**For example:**
```
~/experiments/data/15-10-07/whitenoise.h5
~/experiments/data/15-10-07/naturalscene.h5
```

**If the data doesn't exist, the command won't work.**

See `HOW_TO_RUN.md` for more details on the required data structure.

