import torch


class Model(torch.nn.Module):
    """
    It is a model created by Zepu
    """
    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(Model,self).__init__()
        self.hidden  = 32
        self.size = 3
        self.stride = 2
        self.conv1 = torch.nn.Conv2d(num_channels,self.hidden,kernel_size=(self.size,self.size))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(((32-self.size+1)**2)*self.hidden,num_classes)
        self.pool = torch.nn.MaxPool2d(3,stride = self.stride)
        self.c1 = (32-self.size+1)
        self.c2 = int((self.c1 - 3 + 1) / self.stride)
        self.fc2 = torch.nn.Linear(self.c2 * self.c2 * self.hidden,num_classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x (tensor) : The input data.

        Returns:
            x (tensor): The output of the network.
        """
        x = self.conv1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.pool(x)
        x = self.relu(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x
