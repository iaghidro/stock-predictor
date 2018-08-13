# This script installs Anaconda for current user,
# creates a conda virtual environment named "fast" using the environment.yml file from the fastai repo,
# adds the new environment to jupyter notebook kernel list,
# adds the nbextensions (optional),
# configures jupyter for prompting for password

wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p $HOME/anaconda
export PATH="$HOME/anaconda/bin:$PATH"
echo "export PATH=\"$HOME/anaconda/bin:\$PATH\"" >> ~/.bashrc

# create Anaconda env named "fast" using environment.yml fron fastai repo
conda env create --name fast -f ~/fastai/environment.yml
source activate fast

# add environment to jupyter notebook kernel list
python -m ipykernel install --user --name fast --display-name "fast"

# add nbextensions
conda install -c conda-forge jupyter_contrib_nbextensions

# configure jupyter for prompting for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py
