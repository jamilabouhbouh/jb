# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import fitting_algorithms as fits
import simulations as sim

class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(5): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4))

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)
        S0 = params[:, 3].unsqueeze(1)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0

class Net_abs(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_abs, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(5): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4))

    def forward(self, X):
        params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)
        S0 = params[:, 3].unsqueeze(1)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0

class Net_sig(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_sig, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(5): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4))

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        Dmin = 0
        Dmax = 0.005
        fmin = 0.0
        fmax = 0.7
        Dpmin = 0.005
        Dpmax = 0.5
        S0min = 0.8
        S0max = 1.2

        Dp = Dpmin + torch.sigmoid(params[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
        Dt = Dmin + torch.sigmoid(params[:, 1].unsqueeze(1)) * (Dmax - Dmin)
        Fp = fmin + torch.sigmoid(params[:, 2].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid( params[:, 3].unsqueeze(1)) * (S0max-S0min)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0

class Net_tiny(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_tiny, self).__init__()

        self.b_values_no0 = b_values_no0

        self.fc_layers = nn.ModuleList()
        self.fc_layers.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0)/2)), nn.ELU()])
        self.fc_layers.extend([nn.Linear(round(len(b_values_no0)/2), max(4,round(len(b_values_no0)/3))), nn.ELU()])
        self.fc_layers.extend([nn.Linear(max(4,round(len(b_values_no0)/3)), max(4,round(len(b_values_no0)/4))), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(max(4,round(len(b_values_no0)/4)), 4))

    def forward(self, X):
        params = self.encoder(X) # Dp, Dt, Fp
        Dmin = 0
        Dmax = 0.005
        fmin = 0.0
        fmax = 0.7
        Dpmin = 0.005
        Dpmax = 0.5
        S0min = 0.8
        S0max = 1.2

        Dp = Dpmin + torch.sigmoid(params[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
        Dt = Dmin + torch.sigmoid(params[:, 1].unsqueeze(1)) * (Dmax - Dmin)
        Fp = fmin + torch.sigmoid(params[:, 2].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid( params[:, 3].unsqueeze(1)) * (S0max-S0min)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0

class Net_split(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_split, self).__init__()

        self.b_values_no0 = b_values_no0

        self.fc_layers = nn.ModuleList()
        self.fc_layers2 = nn.ModuleList()
        self.fc_layers3 = nn.ModuleList()
        self.fc_layers4 = nn.ModuleList()
        self.fc_layers5 = nn.ModuleList()

        self.fc_layers2.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0)/2)), nn.ELU()])
        self.fc_layers3.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0)/2)), nn.ELU()])
        self.fc_layers4.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0)/2)), nn.ELU()])
        self.fc_layers5.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0)/2)), nn.ELU()])

        self.fc_layers2.extend([nn.Linear(round(len(b_values_no0)/2), round(len(b_values_no0)/4)), nn.ELU()])
        self.fc_layers3.extend([nn.Linear(round(len(b_values_no0)/2), round(len(b_values_no0)/4)), nn.ELU()])
        self.fc_layers4.extend([nn.Linear(round(len(b_values_no0)/2), round(len(b_values_no0)/4)), nn.ELU()])
        self.fc_layers5.extend([nn.Linear(round(len(b_values_no0)/2), round(len(b_values_no0)/4)), nn.ELU()])

        self.encoder1 = nn.Sequential(*self.fc_layers, *self.fc_layers2, nn.Linear(round(len(b_values_no0)/4), 1))
        self.encoder2 = nn.Sequential(*self.fc_layers, *self.fc_layers3, nn.Linear(round(len(b_values_no0)/4), 1))
        self.encoder3 = nn.Sequential(*self.fc_layers, *self.fc_layers4, nn.Linear(round(len(b_values_no0)/4), 1))
        self.encoder4 = nn.Sequential(*self.fc_layers, *self.fc_layers5, nn.Linear(round(len(b_values_no0)/4), 1))

    def forward(self, X):
        params1 = self.encoder1(X)
        params2 = self.encoder2(X)
        params3 = self.encoder3(X)
        params4 = self.encoder4(X)
        Dmin = 0
        Dmax = 0.005
        fmin = 0.0
        fmax = 0.7
        Dpmin = 0.005
        Dpmax = 0.5
        S0min = 0.8
        S0max = 1.2

        Dp = Dpmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
        Dt = Dmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (Dmax - Dmin)
        Fp = fmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid( params4[:, 0].unsqueeze(1)) * (S0max-S0min)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0


def learn_IVIM(X_train,b_values, arg, batch_size=128, net=None):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    
    S0=np.mean(X_train[:,b_values == 0],axis=1)
    X_train=X_train/S0[:,None]
    if net is None:
        b_values = torch.FloatTensor(b_values[:]).to(device)
        if arg.run_net == 'loss_con':
            net = Net(b_values).to(device)
        elif arg.run_net == 'abs_con':
            net=Net_abs(b_values).to(device)
        elif arg.run_net == 'sig_con':
            net = Net_sig(b_values).to(device)
        elif arg.run_net =='free':
            net = Net(b_values).to(device)
        elif arg.run_net == 'split':
            net = Net_split(b_values).to(device)
        elif arg.run_net == 'tiny':
            net = Net_tiny(b_values).to(device)
        else:
            raise Exception('no valid network was selected')
        # Loss function and optimizer
    else:
        net.to(device)

    criterion = nn.MSELoss().to(device)

    num_batches = len(X_train) // batch_size
    X_train = X_train[:, :] # exlude the b=0 value as signals are normalized
    trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                                    batch_size = batch_size,
                                    shuffle = True,
                                    drop_last = True)
    if arg.optim=='adam':
        optimizer = optim.Adam(net.parameters(), lr=arg.lr, weight_decay=1e-4)
    elif arg.optim=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=arg.lr,momentum=0.9, weight_decay=1e-4)
    elif arg.optim=='adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=arg.lr, weight_decay=1e-4)
    elif arg.optim=='sgdr': #needs some tweaking. The warm restart needs implementing elsewhere
        optimizer = optim.SGD(net.parameters(), lr=arg.lr,momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader))

    # Best loss
    best = 1e16
    num_bad_epochs = 0

    # Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.
        losstotcon = 0.

        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True), 0):
            # zero the parameter gradients
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred, S0pred = net(X_batch)
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            if arg.run_net == 'loss_con':
                loss_con =  (nn.functional.relu(-Dp_pred) + nn.functional.relu(-Dt_pred) + nn.functional.relu(-Fp_pred) \
                           + nn.functional.relu(-S0pred) + nn.functional.relu((Dp_pred-3)) \
                           + nn.functional.relu((Dt_pred-0.05)) + nn.functional.relu((Fp_pred-1)))
                if epoch < 5:
                    losstotcon += 0.0005 * torch.mean(loss_con)
                else:
                    losstotcon += 0.001 * torch.mean(loss_con)
                if epoch < 5:
                    loss = 0.0005 * torch.mean(loss_con)
                else:
                    loss = criterion(X_pred, X_batch) + 0.001 * torch.mean(loss_con)
            else:
                loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            if arg.optim=='sgdr':
                scheduler.step()
            running_loss +=loss.item()
        if arg.optim=='sgdr':
            print('Reset scheduler')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader))
        if arg.run_net == 'loss_con':
            print("Loss: {} of which constrains contributed {}".format(running_loss,losstotcon))
        else:
            print("Loss: {}".format(running_loss))
        # early stopping
        if arg.run_net == 'loss_con':
            if epoch == 6:
                best = 1e16
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    # Restore best model
    net.load_state_dict(final_model)

    return net

def infer_IVIM(data,bvalues,net):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    net.eval()
    S0 = np.mean(data[:,bvalues == 0],axis=1)
    data = data/S0[:,None]
    net.eval()
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                    batch_size = 2056,
                                    shuffle = False,
                                    drop_last = False)
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader),0):
            X_batch=X_batch.to(device)
            _, Dpt, Dtt, Fpt, S0t = net(X_batch)
            S0 = np.append(S0, (S0t.cpu()).numpy())
            Dp = np.append(Dp,(Dpt.cpu()).numpy())
            Dt = np.append(Dt,(Dtt.cpu()).numpy())
            Fp = np.append(Fp,(Fpt.cpu()).numpy())
    if np.mean(Dp)<np.mean(Dt):
        Dp2=Dt
        Dt=Dp
        Dp=Dp2
        Fp=1-Fp
    return [Dp, Dt, Fp, S0]

def isnan(x):
    return x != x

def pretrain(b, arg, SNR=15, net=None, state=1, sims=100000):
    [S,f, D, Dp] = sim.sim_signal(SNR, b, sims = 100000, Dmin = 0.4 /1000, Dmax = 3.5 /1000, fmin = 0.05, fmax = 0.8, Dsmin= 0.05, Dsmax=0.2, rician = False, state=state)
    D = D[:5000]
    Dp = Dp[:5000]
    f = f[:5000]
    meansig=fits.ivim(b, 0.05, 1e-3, 0.25, 1)
    net=learn_IVIM(np.repeat(np.expand_dims(meansig,0),1000,axis=0),b, arg)
    net=learn_IVIM(np.repeat(S[0:100,],10,axis=0),b,arg, net=net)
    net=learn_IVIM(S[0:1000],b,arg,net=net)
    net=learn_IVIM(S,b,arg,net=net)
    torch.save(net.state_dict(), 'results/pretrained_all{net}.pt'.format(net=arg.run_net))
    S=S[:5000]
    paramsNN = infer_IVIM(S, b, net)
    matNN=sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN)
    return net, matNN

def loadnet(nets):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    if nets == 'loss_con':
        net = Net(b_values).to(device)
    elif nets == 'abs_con':
        net=Net_abs(b_values).to(device)
    elif nets == 'sig_con':
        net = Net_sig(b_values).to(device)
    elif nets =='free':
        net = Net(b_values).to(device)
    net.load_state_dict(torch.load('results/pretrained_all{net}.pt'.format(net=nets)))
    net.to(device)
    return net