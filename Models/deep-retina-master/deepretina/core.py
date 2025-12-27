"""
Core tools for training models.
"""
import os
from datetime import datetime
import deepdish as dd
try:
    import keras.callbacks as cb
    from keras.layers import Input
    from keras.models import load_model
    from keras.optimizers import Adam
except ImportError:
    import tensorflow.keras.callbacks as cb
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
from deepretina import metrics, activations
from deepretina.experiments import loadexpt, CELLS

__all__ = ['train', 'load']


def load(filepath):
    """Reload a keras model"""
    objects = {k: activations.__dict__[k] for k in activations.__all__}
    objects.update({k: metrics.__dict__[k] for k in metrics.__all__})
    return load_model(filepath, custom_objects=objects)


def train(model, expt, stim, model_args=(), lr=1e-2, bz=5000, nb_epochs=500, val_split=0.05, cells=None):
    """Trains a model"""
    if cells is None:
        width = None
        cells = CELLS[expt]
        cellname = ''
    else:
        width = 11
        cellname = f'cell-{cells[0]+1:02d}'

    # load experimental data
    data = loadexpt(expt, cells, stim, 'train', 40, 6000, cutout_width=width)

    # build the model
    n_cells = data.y.shape[1]
    x = Input(shape=data.X.shape[1:])
    mdl = model(x, n_cells, *model_args)

    # compile the model
    # For newer Keras versions, use learning_rate parameter
    try:
        optimizer = Adam(learning_rate=lr)
    except TypeError:
        # Fallback for older Keras versions
        optimizer = Adam(lr=lr)
    mdl.compile(loss='poisson', optimizer=optimizer, metrics=[metrics.cc, metrics.rmse, metrics.fev])

    # store results in this directory
    name = '_'.join([mdl.name, cellname, expt, stim, datetime.now().strftime('%Y.%m.%d-%H.%M')])
    base = f'../results/{name}'
    os.makedirs(base, exist_ok=True)

    # define model callbacks
    # TensorBoard parameters changed in newer versions
    try:
        # Try newer TensorBoard API (without write_grads which was deprecated)
        tensorboard_cb = cb.TensorBoard(log_dir=base, histogram_freq=1)
    except TypeError:
        # Fallback for older versions
        tensorboard_cb = cb.TensorBoard(log_dir=base, histogram_freq=1, batch_size=5000, write_grads=True)
    
    cbs = [cb.ModelCheckpoint(os.path.join(base, 'weights-{epoch:03d}-{val_loss:.3f}.h5')),
           tensorboard_cb,
           cb.ReduceLROnPlateau(min_lr=0, factor=0.2, patience=10),
           cb.CSVLogger(os.path.join(base, 'training.csv')),
           cb.EarlyStopping(monitor='val_loss', patience=20)]

    # train
    history = mdl.fit(x=data.X, y=data.y, batch_size=bz, epochs=nb_epochs,
                      callbacks=cbs, validation_split=val_split, shuffle=True)
    dd.io.save(os.path.join(base, 'history.h5'), history.history)

    return history
