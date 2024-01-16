#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=mobo_install


### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=out_mobo_install.%J

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=02:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4000
### Request number of CPUs
#SBATCH --cpus-per-task=3

### Change to the work directory
cd $HPCWORK/DGEMO
source $HOME/.zshrc
micromamba env create -f ./env_man.yml -y
micromamba activate mobo
pip install pygco
python ./main.py --problem k1 --algo tsemo

