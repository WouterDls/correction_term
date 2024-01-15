# A multidimensional AI-trained correction term for the 1D approximation with Gaussian process regression for Airborne TDEM data

**_More details can be found in the manuscript (https://arxiv.org/abs/2311.13998), currently in preparation for review._**

Full 3D inversion of time-domain electromagnetic data requires immense computational resources. To overcome the time-consuming 3D simulations, we propose a surrogate model trained on 3D simulation data and predicts the approximate output much faster. We exemplify the approach for two-layered models.

We construct a surrogate model that predicts the discrepancy between a 1D two-layered subsurface model (low fidelity data, LF) and a deviation of the 1D assumption (high fidelity data, HF).
$$ RE = \frac{HF - LF}{LF} $$
The LF data is efficiently computed with a semi-analytical 1D forward model. The HF data is generated with, e.g., SimPEG (https://simpeg.xyz/).

The results are encouraging even with few training samples, but obtaining a high accuracy is difficult with relatively simple data fit models. We view the performance as a learning gain, representing the gain from the surrogate model while acknowledging a residual discrepancy. 

