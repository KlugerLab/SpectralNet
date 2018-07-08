# SpectralNet
![cc](https://user-images.githubusercontent.com/9156971/34493923-1abbabe8-efbc-11e7-8788-66c62bc91f4d.png)

SpectralNet is a python library that performs spectral clustering with deep neural networks.

## requirements
To run SpectralNet, you'll need Python 3.x and the following python packages:
- scikit-learn
- tensorflow
- munkres
- annoy
- h5py

You will also need wget to download the Reuters dataset, which for MacOS can be installed with
```bash
brew install wget
```

## downloading and preprocessing reuters
To run SpectralNet on the Reuters dataset, you must first download and preprocess it. This can be done by
```bash
cd path_to_spectralnet/data/reuters/; ./get_data.sh; python make_reuters.py
```

## usage
To use SpectralNet on MNIST, Reuters, the nested 'C' dataset (as seen above), or the semi-supervised and noisy nested 'C' dataset, please run
```bash
cd path_to_spectralnet/src/applications; python run.py --gpu=gpu_num --dset=mnist|reuters|cc|cc_semisup
```
To use SpectralNet on a new dataset, simply pass a tuple to get_data (a function in src/core/data.py) containing four elements in the following order: (x_train, x_test, y_train, y_test). Then define the appropriate hyperparameters and call spectralnet.run(). (See example)

## example script
```python
import sys, os
# add directories in src/ to path
sys.path.insert(0, 'path_to_spectralnet/src/')

# import run_net and get_data
from spectralnet import run_net
from core.data import get_data

# define hyperparameters
params = {
    'dset': 'new_dataset',
    'val_set_fraction': ...,
    'siam_batch_size': ...,
    'n_clusters': ...,
    'affinity': ...,
    'n_nbrs': ...,
    'scale_nbrs': ...,
    'siam_k': ...,
    'siam_ne': ...,
    'spec_ne': ...,
    'siam_lr': ...,
    'spec_lr': ...,
    'siam_patience': ...,
    'spec_patience': ...,
    'siam_drop': ...,
    'spec_drop': ...,
    'batch_size': ...,
    'siam_reg': ...,
    'spec_reg': ...,
    'siam_n': ...,
    'siamese_tot_pairs': ...,
    'arch': [
        {'type': 'relu', 'size': ...},
        {'type': 'relu', 'size': ...},
        {'type': 'relu', 'size': ...},
        ],
    'use_approx': ...,
    }
    
# load dataset
x_train, x_test, y_train, y_test = load_new_dataset_data()
new_dataset_data = (x_train, x_test, y_train, y_test)

# preprocess dataset
data = get_data(params, new_dataset_data)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)
```
For more information on what each hyperparameter means, see src/applications/run.py
