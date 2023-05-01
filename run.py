from utils import *
from model import RModel
import torch
from dataloader import ImageLoader
from torch.utils.data import DataLoader
import torchvision
import argparse
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
if not os.path.exists("plots"):
    os.mkdir("plots")

# Evaluation and training functions functions
def get_evaluation_metrics(preds,targets):
    # calculate specificity
    # sepcificity = Correct Label[0] + Correct Label[1] + Correct Label[2] + Correct Label[3] / Total Label[0] + Total Label[1] + Total Label[2] + Total Label[3]
    sensitivity_num = 0
    sensitivity_denm = 0
    for i in range(4):
        sensitivity_num += torch.sum((preds==i) & (targets==i))
        sensitivity_denm += torch.sum(targets==i)
    sensitivity = float(sensitivity_num)/float(sensitivity_denm)
    specificity = float(torch.sum((preds==0) & (targets==0)))/float(torch.sum(targets==0))
    score = 0.5*(sensitivity+specificity)
    return sensitivity,specificity,score

def get_scores(model, testloader, device):
    # model.eval()
    with torch.no_grad():
        val_loss, val_sensitivity, val_specificity, val_score = 0.0, 0.0, 0.0, 0.0
        for val_batch in testloader:
            imgs, targets = val_batch
            imgs, targets = imgs.to(device), targets.to(device).long()
            val_outputs = model(imgs)
            val_loss += torch.nn.CrossEntropyLoss()(val_outputs, targets).item()
            val_preds = torch.argmax(val_outputs, dim=1)
            sensitivity,specificity,score = get_evaluation_metrics(val_preds,targets)
            val_sensitivity += sensitivity
            val_specificity += specificity
            val_score += score

    val_loss = float(float(val_loss)/float(len(testloader)*BATCH_SIZE))
    val_sensitivity = float(float(val_sensitivity)/float(len(testloader)))
    val_specificity = float(float(val_specificity)/float(len(testloader)))
    val_score = float(float(val_score)/float(len(testloader)))
    return val_loss,val_sensitivity,val_specificity,val_score



def evaluate(model, testloader, device):
    # model.eval()
    with torch.no_grad():
        val_loss, val_acc = 0.0, 0.0
        for val_batch in testloader:
            imgs, targets = val_batch
            #print(imgs.shape, targets.shape)
            #print(targets)
            imgs, targets = imgs.to(device), targets.to(device).long()
            val_outputs = model(imgs)
            val_loss += torch.nn.CrossEntropyLoss()(val_outputs, targets).item()
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc += torch.sum(val_preds == targets)

    val_acc = float(float(val_acc)/float(len(testloader)*BATCH_SIZE))
    return val_acc,val_loss

def train_evaluate(model, optimizer, trainloader, valloader, num_epochs=10, save_model_name=None, device="cuda"):
    model.to(device)
    model.train()
    train_accuracy_with_epochs = []
    val_accuracy_with_epochs = []
    loss_with_epochs = []
    val_loss_with_epochs = []
    # best_val_acc = 0.0
    best_loss = 10000
    for epoch in range(num_epochs):
        train_loss, train_acc = 0, 0
        print("\nEpoch: ", str(epoch+1), "/", str(num_epochs))

        with tqdm(total=len(trainloader)) as pbar:
            for idx, batch in enumerate(trainloader):
                images, labels = batch
                images, labels = images.to(device), labels.to(device).long()
                preds = model(images)
                loss = torch.nn.CrossEntropyLoss(weight=train_class_weights)(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(torch.argmax(preds, dim=1),labels)
                train_loss += loss.item()
                acc = torch.sum(torch.argmax(preds, dim=1) == labels)
                train_acc += acc
                pbar.set_postfix(Loss='{0:.4f}'.format(loss.item()), Accuracy='{0:.4f}'.format(float(train_acc.item()/(BATCH_SIZE*(idx+1)))))
                pbar.update(1)

            val_acc, val_loss = evaluate(model, valloader, device)
            val_accuracy_with_epochs.append(val_acc)
            val_loss_with_epochs.append(val_loss)
            loss_with_epochs.append(train_loss)
            train_loss = float(float(train_loss)/float(len(trainloader)*BATCH_SIZE))
            print("train_acc:", round(float(float(train_acc)/float(len(trainloader)*BATCH_SIZE)), 4), " val_acc:", round(val_acc, 4))
            train_accuracy_with_epochs.append(round(float(float(train_acc)/float(len(trainloader)*BATCH_SIZE)), 4))
            
            if train_loss <= best_loss:
                best_loss = train_loss
                if save_model_name is not None:
                    torch.save(model.state_dict(), os.path.join("saved_models", save_model_name))
                    print("Model saved at", save_model_name)
                    print("Best train loss:", best_loss)

    return train_accuracy_with_epochs, val_accuracy_with_epochs, model, loss_with_epochs, val_loss_with_epochs

# CODE STARTS HERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get input from parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="ICBHI_final_database", help="path to dataset directory")
parser.add_argument("--train_test_txt", type=str, default="ICBHI_challenge_train_test.txt", help="path to train_test_txt file")
parser.add_argument("--model_type", type=str, default="hybrid", help="model type")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--patient_id", type=str, default="0", help="patient id")
parser.add_argument("--fine_tune", type=bool, default=False, help="fine tune")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--mode", type=str, default="train_test", help="train, test or train_test")
parser.add_argument("--model_path", type=str, default=None, help="path to saved models state dict")

args = parser.parse_args()
dataset_dir = args.dataset_dir
train_test_txt = args.train_test_txt
model_type = args.model_type
lr = args.lr
num_epochs = args.num_epochs
MODE = args.mode
BATCH_SIZE = args.batch_size
MODEL_PATH= args.model_path

if MODE == "test" and MODEL_PATH is None:
    print("Please provide a model path for testing")
    exit()

train_file_df,test_file_df = get_train_test_names(train_test_txt)

train_dataset = ImageLoader(train_file_df,dataset_dir,train_flag=True)
test_dataset = ImageLoader(test_file_df,dataset_dir,train_flag=False) 
train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[int(0.8*len(train_dataset)),len(train_dataset)-int(0.8*len(train_dataset))])

# create a df with audio array, label, start and end
if MODE == "train_test" or "train":
    train_class_counts = {}
    for i, (audio,label) in enumerate(train_dataset):
        label = int(label)
        if label not in train_class_counts.keys():
            train_class_counts[label] = 1
        else:
            train_class_counts[label]+=1

    train_class_prob = {
        0:train_class_counts[0]/len(train_dataset),
        1:train_class_counts[1]/len(train_dataset),
        2:train_class_counts[2]/len(train_dataset),
        3:train_class_counts[3]/len(train_dataset)
    }

    train_class_weights = torch.from_numpy(np.array([1/train_class_counts[i] for i in range(4)]))
    train_class_weights=train_class_weights.to(device).float()

    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)

test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)


model_types = ["vgg16","mobilenet","hybrid"]

if model_type not in model_types:
    print("model type not supported")
    exit()

elif model_type == "mobilenet":
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights)
    new_conv = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.features[0] = new_conv
    mobilenet_in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(mobilenet_in_features,4)
    model = model.to(device)

elif model_type == "vgg16":
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    model.features[0]=torch.nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1)
    vgg_in_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(vgg_in_features,4)
    model = model.to(device)
else:
    model = RModel()
    model = model.to(device)

if MODEL_PATH is not None:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded from",MODEL_PATH)


optimizer = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99),amsgrad=False)
if MODE == "train_test" or "train":
    train_acc,val_acc,model,loss_with_epochs, val_loss_with_epochs = train_evaluate(model,optimizer,train_loader,val_loader,num_epochs=num_epochs,save_model_name=f"{model_type}_model.pt",device=device)

    plt.figure()
    plt.plot(val_acc,label="Validation")
    plt.plot(train_acc,label="Train")
    plt.legend()
    plt.title(f"Training vs Validation Accuracy for {model_type.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f"plots/{model_type.upper()}_train_vs_validation_acc.png")

    plt.figure()
    plt.plot(loss_with_epochs,label="Train")
    plt.plot(val_loss_with_epochs,label="Validation")
    plt.title(f"{model_type}-Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plots/{model_type}_train_vs_val_loss.png")

if MODE == "train_test" or "test":
    print("--------------------------------------------------------------")
    print("Results on Testing Data")
    loss,specificity, sensitivity, score = get_scores(model, test_loader, device)
    print(f"Test Loss on {model_type.upper()} model: ",loss)
    print(f"Test Specificity on {model_type.upper()} model: ",specificity*100)
    print(f"Test Sensitivity on {model_type.upper()} model: ",sensitivity*100)
    print(f"Test Score on {model_type.upper()} model: ",score*100)
else:
    print("--------------------------------------------------------------")
    print("Results on Training Data")
    loss,specificity, sensitivity, score = get_scores(model, train_loader, device)
    print(f"Train Loss on {model_type.upper()} model: ",loss)
    print(f"Train Specificity on {model_type.upper()} model: ",specificity*100)
    print(f"Train Sensitivity on {model_type.upper()} model: ",sensitivity*100)
    print(f"Train Score on {model_type.upper()} model: ",score*100)
    print("--------------------------------------------------------------")
    print("Results on Validation Data")
    loss,specificity, sensitivity, score = get_scores(model, val_loader, device)
    print(f"Validation Loss on {model_type.upper()} model: ",loss)
    print(f"Validation Specificity on {model_type.upper()} model: ",specificity*100)
    print(f"Validation Sensitivity on {model_type.upper()} model: ",sensitivity*100)
    print(f"Validation Score on {model_type.upper()} model: ",score*100)
    print("--------------------------------------------------------------")


if model_type == "hybrid" and args.fine_tune:
    # Fine Tuning on Patient Wise Data
    patient_wise_tune_data = train_dataset.dataset.patient_to_idx
    # patient to samples is a dict containing patient id as key and list of samples as value
    # we need to fine tune the model patient wise
    # we will use the same model as before 
    #  This code section needs improvement
    patient_id = args.patient_id
    patient_samples = patient_wise_tune_data[patient_id]
    patient_train_samples = patient_samples[:int(0.8*len(patient_samples))]
    patient_test_samples = patient_samples[int(0.8*len(patient_samples)):]
    patient_train_dataset = torch.utils.data.Subset(train_dataset,patient_train_samples)
    patient_test_dataset = torch.utils.data.Subset(train_dataset,patient_test_samples)
    patient_train_loader = torch.utils.data.DataLoader(patient_train_dataset,batch_size=32,shuffle=True)
    patient_test_loader = torch.utils.data.DataLoader(patient_test_dataset,batch_size=32,shuffle=True)

    fine_tune_optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.99),amsgrad=False,weight_decay=0.0001)
    fine_tune_train_acc,fine_tune_val_acc,fine_tune_model,fine_tune_loss_with_epochs,fine_tune_val_loss_with_epochs = train_evaluate(model,fine_tune_optimizer,patient_train_loader,patient_test_loader,num_epochs=num_epochs,save_model_name="fine_tune_model.pt",device=device)
    
    print(f"Results on patient {patient_id} using fine tuned patient specific model")
    loss,specificity, sensitivity, score = get_scores(fine_tune_model, patient_test_loader, device)
    print(f"Test Loss for patient {patient_id} on fine tuned patient specific model: ",loss)
    print(f"Test Specificity for patient {patient_id} on fine tuned patient specific model: ",specificity*100)
    print(f"Test Sensitivity for patient {patient_id} on fine tuned patient specific model: ",sensitivity*100)
    print(f"Test Score for patient {patient_id} on fine tuned patient specific model: ",score*100)
    print(f"-------------------------------------------")
