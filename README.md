# ts_windowless_transformer

This repo has a sklearn compatible transformer that transforms multi variate time series into tabular form.

The transformer takes each time series in the multivariate time series, computes DCT-2 transformation on multiple subwindows, takes the most significant coefficients, and concatenates these coefficients to create the final tabular data.

How to run:
python ts_transformer.py

Expected output:
UTS Transformer, Xt: (100, 9)
MTS Transformer, Xt: (100, 18)


