import pickle as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial, Fourier
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

def get_data(fname):
    fname = f"./data/{fname}"
    with open(f"{fname}_ref", 'rb') as f:
        inp_sig = p.load(f)
    with open(fname, 'rb') as f:
        data = p.load(f)
    return data, inp_sig

def normalize_data(data):
    data = data.reshape(-1,1)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

data, ref_signal = get_data("Two_blocks_900Hz")

y = normalize_data(data[0])
X = normalize_data(ref_signal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# '''Create some samples with Gaussian noise'''
# x_train, x_valid, y_train, y_valid = get_siso_data(
#     n=1000,
#     colored_noise=False,
#     sigma=0.0001,
#     train_percentage=90)

'''Model selection'''
basis_function = Polynomial(degree=2)
# basis_function = Fourier(degree=2, n=2, p=2*np.pi, ensemble=True)

'''Default model from documentation'''
model = FROLS(
    order_selection=True,
    n_info_values=3,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
    basis_function=basis_function
)

'''Fit and predict'''
model.fit(X=X_train, y=y_train)
yhat = model.predict(X=X_test, y=y_test)

'''Evaluate'''
rrse = root_relative_squared_error(y_test, yhat)
print(rrse)
r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

plot_results(y=y_test, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_test, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_test, yhat, X_test)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")

