#!/usr/bin/env bash
#SBATCH --job-name=RespiratorySoundClassification
#SBATCH --output=RespiratorySoundClassification/respiratory_sound_classification.out
#SBATCH --error=RespiratorySoundClassification/respiratory_sound_classification.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

#!/bin/bash
# # Set up the default python path
export PYTHONPATH=$PYTHONPATH:/home/usr/bin/python3 #set your corresponding python path here

# # Activate the virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt


python3  run.py --lr=0.0001 --model_type=mobilenet --num_epochs=10 --batch_size=32 # To train the mobilenet backbone based model
python3  run.py --lr=0.0001 --model_type=vgg16 --num_epochs=10 --batch_size=32 # To train the Vgg16 backbone based model
python3  run.py --lr=0.0001 --model_type=hybrid --num_epochs=10 --batch_size=32 # To train the Hybrid CNN RNN model
python3  run.py --lr=0.0001 --model_type=hybrid --num_epochs=10 --patient_id=107 --fine_tune=True --batch_size=32
