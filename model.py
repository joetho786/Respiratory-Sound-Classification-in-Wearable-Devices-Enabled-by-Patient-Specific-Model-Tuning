from torch.utils.data import Dataset, DataLoader
import torch
import os

class RModel(torch.nn.Module):
    def __init__(self):
        super(RModel,self).__init__()
        self.hidden_dim = 256
        self.num_layers = 1
        self.input_dim = 128
        self.fc_dim = 36864
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            # conv layer with kernel 5x7
            torch.nn.Conv2d(1, 64, kernel_size=(5,7)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,3)),
            torch.nn.Conv2d(64,64,kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,3)),
            torch.nn.Conv2d(64,128,kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2,3)),
            torch.nn.Conv2d(128,128,kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,kernel_size=(3,3)),
            torch.nn.ReLU(),
        )
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.fc_dim,100),
            # torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100,4),
            torch.nn.Softmax()
        )
        self.bi_lstm_block = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
    
    
    def init_hidden_state(self, batch_size,device='cuda'):
        return (torch.zeros((self.num_layers * 2, batch_size, self.hidden_dim),device=device),
                torch.zeros((self.num_layers * 2, batch_size, self.hidden_dim),device=device))

    def init_optimizer(self,learning_rate=0.0001):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, amsgrad=False,betas=(0.9, 0.999), weight_decay=5e-4)
    
    def forward(self, x):
        # reshape the input to (batch_size, num_channels, height, width)
        # x = x.permute(0, 3, 2, 1)
        x = self.feature_extractor(x)
        batch_size, C, H, W = x.size()
        x = x.view(batch_size, C, H*W)
        x = x.permute(0, 2, 1)
        self.fc_dim = self.hidden_dim * H * W * 2
        h0, c0 = self.init_hidden_state(batch_size)
        x, (hn, cn) = self.bi_lstm_block(x, (h0, c0))
        x = torch.nn.Tanh()(x)
        x = self.fully_connected(x)
        return x

    
    


def save_model(model,name='model.pt'):
    if not os.path.isdir('saved_models'):
        os.mkdir('saved_models')
      
    torch.save(model.state_dict(), os.path.join('saved_models', name))
    print("Model successfully saved.")
    
def load_model(model):
    model.load_state_dict(torch.load(os.path.join('saved_models', 'model.pt')))
    return model

