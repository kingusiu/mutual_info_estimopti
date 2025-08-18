import torch.nn as nn

activation_functions = {
    'elu': nn.ELU(),
    'relu': nn.ReLU(),
    'sigm': nn.Sigmoid(),
    'tanh': nn.Tanh(),
}

class MLP_Surrogate(nn.Module):
    """MLP surrogate model for estimating mutual information from design parameters.
    Args:
        N_feat (int): Number of input features.
        acti (str): Activation function to use in the hidden layers.
        N_hidden (int): Number of neurons in the hidden layers.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    
    def __init__(self, N_feat=1, acti='elu', N_hidden=10, device='cpu'):

        self.device = device

        super().__init__()
        self.layer1 = nn.Linear(N_feat, N_hidden)
        self.bn1 = nn.BatchNorm1d(N_hidden)
        self.act1 = activation_functions[acti.lower()]
        self.layer2 = nn.Linear(N_hidden, N_hidden*2)
        self.bn2 = nn.BatchNorm1d(N_hidden*2)
        self.act2 = activation_functions[acti.lower()]
        self.layer3 = nn.Linear(N_hidden*2, N_hidden)
        self.bn3 = nn.BatchNorm1d(N_hidden)
        self.act3 = activation_functions[acti.lower()]
        self.output = nn.Linear(N_hidden, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.layer1(x)))
        x = self.act2(self.bn2(self.layer2(x)))
        x = self.act3(self.bn3(self.layer3(x)))
        x = self.output(x)
        return x

def train_surrogate(surr_model, surr_opt, surr_metric, dataloader, data_valid, scheduler, n_epochs):

    losses_train = []
    losses_valid = []

    ### training

    for _ in range(n_epochs):
        epoch_loss = 0
        surr_model.train()

        for i, batch in enumerate(dataloader):
            theta, mi = [b.to(surr_model.device) for b in batch]
            surr_opt.zero_grad()
            output = surr_model(theta)
            loss = surr_metric(output, mi)
            loss.backward()
            surr_opt.step()
            scheduler.step()
            epoch_loss += loss.item()

        losses_train.append(epoch_loss / len(dataloader) / dataloader.batch_size) # loss per sample


        ### validation

        surr_model.eval()
        mi_valid = surr_model(data_valid.theta.to(surr_model.device))
        losses_valid.append(surr_metric(mi_valid, data_valid.mi.to(surr_model.device)).item() / len(data_valid)) # loss per sample

    return losses_train, losses_valid
