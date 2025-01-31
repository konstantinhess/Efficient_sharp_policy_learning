Efficient and sharp policy learning under unobserved confounding
==============================

We develop an efficient and sharp estimator for off-policy learning under unobserved confounding using the marginal sensitivity model.

### Setup
Please set up a virtual environment and install the libraries as given in the requirements file.
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow
To start an experiments server, run: 

`mlflow server --port=3336`

Connect via ssh to access the MlFlow UI:

`ssh -N -f -L localhost:3336:localhost:3336 <username>@<server-link>`

Then, one can go to the local browser <http://localhost:3336>.

## Experiments

The config file can be found under `config/config.yaml`. 

To replicate our experiments, use the configurations as given in the config file. All experiments are run over seeds `exp.global_seed=0,1,2...,9`.

Note that, before running experiments on RCT stroke data, the .csv file (https://pmc.ncbi.nlm.nih.gov/articles/PMC3104487/) needs to be placed in `./data/` and processed via `./notebooks/rwd_preprocessing`.

___

