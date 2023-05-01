# Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning
## Introduction
This repository contains the UNOFFICIAL code implementation for the paper "Deep Neural Network for Respiratory Sound Classification in Wearable Devices Enabled by Patient Specific Model Tuning" by Jyotibdha Acharya , Student Member, IEEE, and Arindam Basu , Senior Member, IEEE. The paper is available ![here][https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9040275].

## Instructions to Setup
1. Clone the repository
2. Download the dataset from ![here][https://www.kaggle.com/vbookshelf/respiratory-sound-database]
3. Extract the dataset and place it in the same directory as the repository
4. Run the train.sh script to train the all the models or
5. Run the following command to train a specific model
```bash
python run.py --dataset_dir=<path to dataset> --mode=<train,test,or train_test> --train_test_txt=<path to txt file containing train test ids> --batch_size=<batch_size> --num_epochs=<no of epochs> --lr=<learning rate> --model_path=<path to model state dic.Note in such case model_type must be mentioned> --model_type=<vgg16,mobilenet,hybrid> --fine_tune=<True or False. In this case patient_id is compulsory> --patient_id=<patient id> 
```
Here the arguments are:
* dataset_dir: Path to the dataset
* mode: train,test,or train_test
* train_test_txt: Path to the txt file containing train test ids
* batch_size: Batch size
* num_epochs: Number of epochs
* lr: Learning rate
* model_path: Path to model state dic.Note in such case model_type must be mentioned
* model_type: vgg16,mobilenet,hybrid
* fine_tune: True or False. In this case patient_id is compulsory
* patient_id: Patient id

Results and code sample can also be seen from the demo ipynb in the repo
