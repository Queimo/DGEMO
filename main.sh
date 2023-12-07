#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=mobo
#### SBATCH --account=rwth1430


### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=training-out_mobo.%J

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=08:00:00

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4000
### Request number of CPUs
#SBATCH --cpus-per-task=8

### Change to the work directory
cd $HPCWORK/DGEMO
source $HOME/.zshrc
source activate mobo

python ./main.py $@