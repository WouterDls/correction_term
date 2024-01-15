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
import warnings as wa
times = np.logspace(-5, -2, 100)


"""
    IMPORT DATA
"""

X = pd.read_csv('parameters.csv').drop(columns='Unnamed: 0').to_numpy() # Input parametrs: depths, angles, EC0, EC1
HF = pd.read_csv('dataHF.csv').drop(columns='Unnamed: 0').to_numpy() # High fidelity data via simulations for parameters in X
LF = pd.read_csv('dataLF.csv').drop(columns='Unnamed: 0').to_numpy() # Low fidelity data via analytical models for parameters in X.


### Data transformations
def data_transform(x):
    """
    Data transformation on the electrical conductivities.
    .. math::
    	\tilde{EC} = a + \frac{(b-a)\left(\log_{10}EC - \log_{10}(\min EC\right)}{\log_{10}\max(EC) - \log_{10}\min(EC)}
    :param x: input param, electrical conductivity in S/m
    :return: log-like transformed electrical conductivity
    """
    a, b = -1, 1
    return a + (b - a) * (np.log10(x) - np.log10(np.min(x))) / (np.log10(np.max(x)) - np.log10(np.min(x)))

min_x = np.min(X[:, 2])
max_x = np.max(X[:, 2])

def inverse_data_transform(x, min_x, max_x):
    """
        Inverse data transformation on the electrical conductivities.
        .. math::
        	\tilde{EC} = a + \frac{(b-a)\left(\log_{10}EC - \log_{10}(\min EC\right)}{\log_{10}\max(EC) - \log_{10}\min(EC)}
        :param x: log-like transformed electrical conductivity
        :return: the original electrical conductivity in S/m
        """
    a, b = -1, 1
    return 10 ** (x * (np.log10(max_x) - np.log10(min_x) / ((b - a) * (x - np.log10(min_x)))))

# Performing the data transformation on the electrical conductivities
X[:, 2] = data_transform(X[:, 2])
X[:, 3] = data_transform(X[:, 3])

## Generating test and training dataset
"""
1. Transformation to relative errors RE
2. Splitting in training set and test set
"""
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

# Prepped parameters and data
X_prep = feature_pipeline.fit_transform(X_train)
y_prep = target_pipeline.fit_transform(y_train)

xtest = feature_pipeline.transform(X_test)
ytest = target_pipeline.transform(y_test)

parameterspace = PCA()
parameterspace.fit(X_prep)

## Training the separate surrogate models (per PCA-component)
surrogates = []

#  pre-optimized hyperparameters
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

n_components = 8  # optimal number of components to retain (see manuscript)

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

##
def check_for_extrapolation(x):
    """
    Functionality throws warnings if the parameters are extrapolating (or at the edge of the trained parameter space).
    :param x: input parameters for the surrogate model, after transformation
    :return: None
    """
    x0 = -0.25
    y0 = -0.25
    r = 1.7

    if x.size == 4:
        x_pca = parameterspace.transform(x.reshape(1,-1))

        if ((x_pca[:, 0] - x0) ** 2 + (x_pca[:, 1] - y0) ** 2) > r ** 2:
            wa.warn('Extrapolation warning: The provided input parameters are outside of the stable trained parameter space.')
    else:
        for idx, xx in enumerate(x):
            x_pca = parameterspace.transform(xx.reshape(1, -1))
            if ((x_pca[:, 0] - x0) ** 2 + (x_pca[:, 1] - y0) ** 2) > r ** 2:
                wa.warn(
                    'Extrapolation warning for sample ' + str(idx)  + ': The provided input parameters are outside of the stable trained parameter space.')


def make_prediction(x, surrogates):
    """

    :param x: input parameters for the surrogate model, after transformation
    :param surrogates: list of SISO GPR models as models
    :return: The prediction of the surrogate model, in fca-space.
    """
    check_for_extrapolation(x)
    mean, = np.zeros((x.shape[0], len(surrogates))),
    for idx, surrogate in enumerate(surrogates):
        mean[:, idx] = surrogate.predict(x, return_std=False)
    return mean

ypred = make_prediction(xtest, surrogates)

##
"""
Examles 
"""
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



