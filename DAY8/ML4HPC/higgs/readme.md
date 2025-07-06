Based on https://docs.rapids.ai/deployment/stable/examples/xgboost-dask-databricks/notebook/

Dataset available at https://archive.ics.uci.edu/dataset/280/higgs

The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level features then 7 high-level features):

- lepton  pT, 
- lepton  eta,
- lepton  phi,
- missing energy magnitude,
- missing energy phi,
- jet 1 pt,
- jet 1 eta,
- jet 1 phi,
- jet 1 b-tag,
- jet 2 pt,
- jet 2 eta,
- jet 2 phi,
- jet 2 b-tag,
- jet 3 pt,
- jet 3 eta,
- jet 3 phi,
- jet 3 b-tag,
- jet 4 pt,
- jet 4 eta,
- jet 4 phi,
- jet 4 b-tag,
- m_jj,
- m_jjj,
- m_lv,
- m_jlv,
- m_bb,
- m_wbb,
- m_wwbb

For more detailed information about each feature see the original paper:

Searching for Exotic Particles in High-Energy Physics with Deep Learning by Pierre Baldi, Peter Sadowski, Daniel Whiteson [ https://arxiv.org/abs/1402.4735 ]


