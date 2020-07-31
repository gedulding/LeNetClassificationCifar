import torch;
from torch import nn;
from torch.nn import functional as F;

class leNet5(nn.Module):
    def __init__(self):
        super(leNet5 , self).__init__();
        self.conv_unit =  nn.Sequential(
            # x : [b, 3, 32, 32] => [b, 16, 5, 5]
            # conv2d(channel , filter_number , kernel_size);
            nn.Conv2d(3 , 6 , kernel_size = 5 , stride = 1 , padding = 0),
            nn.MaxPool2d(kernel_size = 2 , stride = 2 , padding = 0),
            nn.Conv2d(6 , 16 , kernel_size = 5 , stride = 1 , padding = 0),
            nn.MaxPool2d(kernel_size = 2 , stride = 2 , padding = 0),
        )

        #flatten
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5 , 120),
            nn.ReLU(),
            nn.Linear(120 , 84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self , x):
        batchsize = x.size(0)
        # x : [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x);
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.reshape(batchsize , 16*5*5);
        #[b, 16*5*5] => [b, 10]
        labels = self.fc_unit(x);
        return labels;

def main():
    net = leNet5();

if __name__ == '__main__':
    main();