import torch.nn as nn

class MF_CNN(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(MF_CNN, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn_liu(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn_liu(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn_liu(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*hidden_size , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = self.conv3(x)
        
        x = x.view(x.shape[0] ,-1)
        
        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk





    
    
class cnn_liu(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn_liu, self).__init__()
        
        self.cnn = nn.Conv1d(in_channel , hidden_channel , kernel_size = 5 , stride = 1) # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size = 2 , stride=2)# bs * 32*30
                               
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        #x = self.emb(x)
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x