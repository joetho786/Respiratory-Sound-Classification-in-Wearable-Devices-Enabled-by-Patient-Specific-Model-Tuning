# Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning
## Introduction
This repository contains the UNOFFICIAL code implementation for the paper "Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning" by Jyotibdha Acharya , Student Member, IEEE, and Arindam Basu , Senior Member, IEEE. The paper is available [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9040275)

## Instructions to Setup
1. Clone the repository
2. Download the dataset from [here](https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip). You would need to extract both the text files and the audio file including the ICBHI_challenge_train_test.txt files
3. Extract the dataset and place it in the same directory as the repository
4. Run the run.sh script to train the all the models using the following command
```bash
bash run.sh
```
or 
5. Run the following command to train a specific model
```bash
python run.py --dataset_dir=<path to dataset> --mode=<train,test,or train_test> --train_test_txt=<path to txt file containing train test ids> --batch_size=<batch_size> --num_epochs=<no of epochs> --lr=<learning rate> --model_path=<path to model state dic.Note in such case model_type must be mentioned> --model_type=<vgg16,mobilenet,hybrid> --fine_tune=<True or False. In this case patient_id is compulsory> --patient_id=<patient id> 
```
Here the arguments are:
* dataset_dir: Path to the dataset | default: ./ICBHI_final_database
* mode: train,test,or train_test | default: train_test
* train_test_txt: Path to the txt file containing train test ids | default: ./ICBHI_challenge_train_test.txt
* batch_size: Batch size | default: 32
* num_epochs: Number of epochs | default: 100
* lr: Learning rate | default: 0.0001
* model_path: Path to model state dict.Note in such case model_type must be mentioned
* model_type: vgg16,mobilenet,hybrid | default: hybrid
* fine_tune: True or False. In this case patient_id is compulsory | default: False
* patient_id: Patient id | default: 0


Results and code sample can also be seen from the demo ipynb in the repo

## Reference
All the code is written by referring to the research paper:
```
@ARTICLE{9040275,
  author={Acharya, Jyotibdha and Basu, Arindam},
  journal={IEEE Transactions on Biomedical Circuits and Systems}, 
  title={Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning}, 
  year={2020},
  volume={14},
  number={3},
  pages={535-544},
  doi={10.1109/TBCAS.2020.2981172}}
```