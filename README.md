# RobustInference_RHGI

Code for the paper **Robust inference for risk heterogeneity under group imbalance**.

arXiv: [https://arxiv.org/abs/2606.00797](https://arxiv.org/abs/2606.00797)

## Repository structure

| Folder | Contents |
| --- | --- |
| `scenario1/` | Simulation code for Scenario 1 (linear setting): simulation script, cluster job scripts, plotting and QQ-plot code. |
| `scenario2/` | Simulation code for Scenario 2 (nonlinear setting): simulation script, cluster job scripts, plotting and QQ-plot code. |
| `real data/` | Code for the eICU data application (`source.R`, `target.R`, `test.R`). |

## Data availability

The data analysed in this study are available from the eICU Collaborative Research Database, version 2.0 (Pollard et al. 2018, 2019), hosted on PhysioNet (Goldberger et al. 2000) at <https://doi.org/10.13026/C2WM1R>. Access requires a credentialed PhysioNet account and a signed data use agreement; details are available at <https://eicu-crd.mit.edu/gettingstarted/access/>.

Because of the eICU data use agreement, the following files are **not** included in this repository:

- `real data/alldata_cleaned.csv`: the cleaned eICU analysis dataset
- `real data/results/`: results produced from the eICU data
- `real data/source_model.rds`: the source model fitted on the eICU data (regenerate it with `source.R`)

To reproduce the data application, get access to eICU-CRD through PhysioNet, build the cleaned dataset `alldata_cleaned.csv`, put it in `real data/`, and run the scripts in that folder.
