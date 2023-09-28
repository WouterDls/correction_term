import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skfda
from sklearn import (
    model_selection,
    pipeline,
    preprocessing,
)
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.base import BaseEstimator, TransformerMixin

times = np.logspace(-5, -2, 100)


"""
    IMPORT DATA
"""
import numpy as np

X = pd.read_csv('parameters.csv').drop(columns='Unnamed: 0').to_numpy()
HF = pd.read_csv('dataHF.csv').drop(columns='Unnamed: 0').to_numpy()
LF = pd.read_csv('dataLF.csv').drop(columns='Unnamed: 0').to_numpy()


### Data transformation
def data_transform(x):
    a, b = -1, 1
    return a + (b - a) * (np.log10(x) - np.log10(np.min(x))) / (np.log10(np.max(x)) - np.log10(np.min(x)))


min_x = np.min(X[:, 2])
max_x = np.max(X[:, 2])


def inverse_data_transform(x, min_x, max_x):
    a, b = -1, 1
    return 10 ** (x * (np.log10(max_x) - np.log10(min_x) / ((b - a) * (x - np.log10(min_x)))))


X[:, 2] = data_transform(X[:, 2])
X[:, 3] = data_transform(X[:, 3])

## Generating test and training dataset

Y = (HF - LF) / LF

sets = model_selection.train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = sets

## Dimensionality reduction
n_basis = 16
n_components = 8

feature_pipeline = pipeline.Pipeline([
    ('std_scaler', preprocessing.StandardScaler()),  # Preprocessing
])

class fdaTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_basis):
        self.n_basis = n_basis

    def fit(self, y):
        return self  # nothing else to do

    def transform(self, y):
        fd = skfda.FDataGrid(y, np.log10(times))
        basis = skfda.representation.basis.BSpline(n_basis=self.n_basis)
        return fd.to_basis(basis, ).coefficients

    def inverse_transform(self, y):
        basis = skfda.representation.basis.BSpline(n_basis=self.n_basis)
        ypredict_dbasis = skfda.representation.basis.FDataBasis(basis, y)
        ypredict_dgrid = ypredict_dbasis.to_grid(grid_points=np.linspace(0, 1, 100))
        return np.squeeze(ypredict_dgrid.data_matrix)


# Target
target_pipeline = pipeline.Pipeline([
    ('fda', fdaTransform(n_basis=n_basis)),
    ('pca', PCA(n_components=n_components)),
    ('std_scaler', preprocessing.StandardScaler()),  # Preprocessing

])

X_prep = feature_pipeline.fit_transform(X_train)
y_prep = target_pipeline.fit_transform(y_train)

xtest = feature_pipeline.transform(X_test)
ytest = target_pipeline.transform(y_test)

## Training the separate surrogate models (per PCA-component)
surrogates = []

# after optimization
lengthscales = np.c_[[0.58783242, 0.42265738, 1.31803686, 1.35952398],
       [0.14528683, 0.51416033, 1.27277555, 2.26447619],
       [0.56263156, 0.34517462, 1.03387712, 1.39177843],
       [0.46706812, 0.39057878, 0.61654401, 1.31384065],
       [0.13128609, 0.39716071, 0.53057227, 1.3141512 ],
       [0.14387723, 0.43028857, 0.58627865, 1.20699364],
       [0.18468373, 0.53663596, 0.42761323, 1.41184322],
       [0.37851199, 0.34860775, 0.49890485, 0.90396628]].T

noiselevels = np.r_[4.95412821e-03, 2.07320627e-03, 1.17698718e-02, 9.55207967e-03,
       2.27040512e-03, 2.08149970e-03, 1.00000000e-05, 1.00000000e-05]

n_components = 8  # See preprint

for i in np.arange(n_components):
    # surrogates.append(GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-2))
    if i < 4:
        alpha = 1e-5
    elif i >= 6:
        alpha = 1e-2
    else:
        alpha = 1e-3
    kernel = RBF(length_scale=lengthscales[i, :],
                 length_scale_bounds='fixed') \
             + WhiteKernel(noise_level=noiselevels[i],
                           noise_level_bounds='fixed')
    surrogates.append(GaussianProcessRegressor(kernel=kernel, alpha=alpha))

## Fitting the models with pre-optimized hyperparameters

for idx, surrogate in enumerate(surrogates):
    surrogate.fit(X_prep, y_prep[:, idx])

def make_prediction(x, surrogates, return_std=False):
    mean, = np.zeros((x.shape[0], len(surrogates))),
    for idx, surrogate in enumerate(surrogates):
        mean[:, idx] = surrogate.predict(x, return_std=False)
    return mean

ypred = make_prediction(xtest, surrogates)

##

sample_idx = 1

plt.plot(ytest[sample_idx, :],'o')
plt.plot(ypred[sample_idx, :],'o')
plt.title('Relative error in reduced data space')
plt.show()

##

plt.semilogx(times, target_pipeline.inverse_transform(ytest[sample_idx, :].reshape(1,-1)))
plt.semilogx(times, target_pipeline.inverse_transform(ypred[sample_idx, :].reshape(1,-1)))
plt.title('Relative error between HF and LF')
plt.show()

##



