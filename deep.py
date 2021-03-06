# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import fitting_algorithms as fits
import simulations as sim
import matplotlib.pyplot as plt

# here I define several neural networks. This one is the vanilla network, with no parameter constraints
class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()
        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers with ELU at the end
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4)) #final linear layer

    def forward(self, X):
        params = self.encoder(X) # estimates of Dp, Dt, Fp and S0
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)
        S0 = params[:, 3].unsqueeze(1)
        #here we give the expected signal decay, given we actually estimated Dp, Dt, Fp and S0
        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))
        #output parameters and estimated signal decay
        return X, Dp, Dt, Fp, S0
# this network constrains the estimated parameters to be positive by taking the absolute. Advantage is that the parameters are constrained and that the derrivative of the function remains constant. Disadvantage is that -x=x, so could become unstable.
# only added comments where the network is different from the one above
class Net_abs(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_abs, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 4))

    def forward(self, X):
        # at this point, the absolute is taken.        
        params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)
        S0 = params[:, 3].unsqueeze(1)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0
# this network constrains the estimated parameters between two values by taking the sigmoid. Advantage is that the parameters are constrained and that the mapping is unique. Disadvantage is that the gradients go to zero close to the aprameter bounds.
# only added comments where the network is different from the one above
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
        # Defining constraints
        Dmin = -0.0004
        Dmax = 0.005
        fmin = -0.05
        fmax = 0.6
        Dpmin = 0.005
        Dpmax = 0.3
        S0min = 0.7
        S0max = 1.3
        #applying constraints
        Dp = Dpmin + torch.sigmoid(params[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
        Dt = Dmin + torch.sigmoid(params[:, 1].unsqueeze(1)) * (Dmax - Dmin)
        Fp = fmin + torch.sigmoid(params[:, 2].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid( params[:, 3].unsqueeze(1)) * (S0max-S0min)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0
# this network is smaller and has less nodes as we go through the network, like a funnel. Sigmoid constraints
# only added comments where the network is different from the one above
class Net_tiny(nn.Module):
    def __init__(self, b_values_no0):
        super(Net_tiny, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        # the network is 3 layers, with each layer smaller than the one before
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
# this network splits the estimation of parameters into different networks, with sigmoid estimates
# only added comments where the network is different from the one above
class Net_split(nn.Module):
    def __init__(self, b_values_no0, arg):
        super(Net_split, self).__init__()
        self.b_values_no0 = b_values_no0
        # here the splitting starts
        #self.shared_layers = nn.Sequential(
        #    nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU(),
        #    nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU(),
        #    nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU())
        self.fc_layers1 = nn.ModuleList()
        self.fc_layers2 = nn.ModuleList()
        self.fc_layers3 = nn.ModuleList()
        self.fc_layers4 = nn.ModuleList()
        # extending the split networks
        #for aa in range(2):
        #  self.shared_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        for aa in range(5):
            self.fc_layers1.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0))), nn.BatchNorm1d(len(b_values_no0)), nn.Dropout(arg.dropout), nn.ELU()])
            self.fc_layers2.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0))), nn.BatchNorm1d(len(b_values_no0)), nn.Dropout(arg.dropout), nn.ELU()])
            self.fc_layers3.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0))), nn.BatchNorm1d(len(b_values_no0)), nn.Dropout(arg.dropout), nn.ELU()])
            self.fc_layers4.extend([nn.Linear(len(b_values_no0), round(len(b_values_no0))), nn.BatchNorm1d(len(b_values_no0)), nn.Dropout(arg.dropout), nn.ELU()])
        self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(round(len(b_values_no0)), 1))
        self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(round(len(b_values_no0)), 1))
        self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(round(len(b_values_no0)), 1))
        self.encoder4 = nn.Sequential(*self.fc_layers4, nn.Linear(round(len(b_values_no0)), 1))
        
    def forward(self, X):
        #X = self.shared_layers(X)
        params1 = self.encoder1(X)
        params2 = self.encoder2(X)
        params3 = self.encoder3(X)
        params4 = self.encoder4(X)
        Dmin = 0
        Dmax = 0.005
        fmin = 0.0
        fmax = 0.7
        Dpmin = 0.005
        Dpmax = 0.7
        S0min = 0.8
        S0max = 1.2

        Dp = Dpmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dpmax - Dpmin)
        Dt = Dmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (Dmax - Dmin)
        Fp = fmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid( params4[:, 0].unsqueeze(1)) * (S0max-S0min)

        X = S0 * (Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt))

        return X, Dp, Dt, Fp, S0

# This program trains the networks
def learn_IVIM(X_train,b_values, arg, net=None):
    # X_train = input training data
    # b_values=b-values used
    # arg are the input arguments, with: 
    # arg.run_net the desired network
    # arg.lr the learning rate
    # arg.optim the optimiser
    # arg.patience the patience
    # batch_size is the size of the batches used for training
    # net is an optional input when using a pre-trained network that you want to retrain
    # load CUDA for PyTorch, if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    # normalise the signal to b=0
    S0=np.mean(X_train[:,b_values == 0],axis=1)
    X_train=X_train/S0[:,None]
    # removing non-IVIM-like data
    X_train=X_train[np.max(X_train,axis=1)<1.5]
    # initialising the network of choice using the input argument arg
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
            net = Net_split(b_values,arg).to(device)
        elif arg.run_net == 'tiny':
            net = Net_tiny(b_values).to(device)
        else:
            raise Exception('no valid network was selected')
        # Loss function and optimizer
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning).
        net.to(device)
    # defining the loss function
    criterion = nn.MSELoss(reduction='mean').to(device)
    # splitting data into learning and validation set; subesequent initialising the Dataloaders
    split=int(np.floor(len(X_train)*0.8))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)), [split, len(X_train)-split])
    trainloader = utils.DataLoader(train_set,
                                    batch_size = arg.batch_size,
                                    shuffle = True,
                                    drop_last = True)
    inferloader = utils.DataLoader(val_set,
                                    batch_size = 8*arg.batch_size,
                                    shuffle = False,
                                    drop_last = True)
    #defining the number of training and validation batches for normalisation later
    totalit=np.min([1000,np.floor(split // arg.batch_size)])
    batch_norm = totalit
    batch_norm2 = np.floor(len(val_set)// (8*arg.batch_size))
    # defining optimiser
    if arg.optim=='adam':
        optimizer = optim.Adam([{'params': net.encoder1.parameters(), 'lr': arg.lr},{'params': net.encoder2.parameters()},{'params': net.encoder3.parameters()},{'params': net.encoder4.parameters()}], lr=arg.lr, weight_decay=1e-4)
    elif arg.optim=='sgd':
        optimizer = optim.SGD([{'params': net.encoder1.parameters(), 'lr': arg.lr},{'params': net.encoder2.parameters()},{'params': net.encoder3.parameters()},{'params': net.encoder4.parameters()}], lr=arg.lr,momentum=0.9, weight_decay=1e-4)
    elif arg.optim=='adagrad':
        optimizer = torch.optim.Adagrad([{'params': net.encoder1.parameters(), 'lr': arg.lr},{'params': net.encoder2.parameters()},{'params': net.encoder3.parameters()},{'params': net.encoder4.parameters()}], lr=arg.lr, weight_decay=1e-4)
    elif arg.optim=='sgdr': #needs some tweaking. The warm restart needs implementing elsewhere
        optimizer = optim.SGD([{'params': net.encoder1.parameters(), 'lr':arg.lr},{'params': net.encoder2.parameters()},{'params': net.encoder3.parameters()},{'params': net.encoder4.parameters()}], lr=arg.lr,momentum=0.9, weight_decay=1e-4)
    if arg.optim == 'sgdr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader))
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train=[]
    loss_val=[]
    bestlist=[]
    bestepochlist=[]
    prev_lr=0
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        #initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_val = 0.
        losstotcon = 0.
        maxloss=0.

        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i>totalit:
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            if i == 3:
                optimizer.param_groups[0]['lr']=arg.lr_Ds
            X_batch = X_batch.to(device)
            ## forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred, S0pred = net(X_batch)
            # removing nans and outragous predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # something I tried in the past with constraints in the loss function (not functional now)
            if arg.run_net == 'loss_con':
                loss_con =  (nn.functional.rELU(-Dp_pred) + nn.functional.rELU(-Dt_pred) + nn.functional.rELU(-Fp_pred) \
                           + nn.functional.rELU(-S0pred) + nn.functional.rELU((Dp_pred-3)) \
                           + nn.functional.rELU((Dt_pred-0.05)) + nn.functional.rELU((Fp_pred-1)))
                if epoch < 5:
                    losstotcon += 0.0005 * torch.mean(loss_con)
                else:
                    losstotcon += 0.001 * torch.mean(loss_con)
                if epoch < 5:
                    loss = 0.0005 * torch.mean(loss_con)
                else:
                    loss = criterion(X_pred, X_batch) + 0.001 * torch.mean(loss_con)
            else:
                # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
                loss = criterion(X_pred, X_batch)
            # updating network
            loss.backward()
            optimizer.step()
            if arg.optim == 'sgdr':
                scheduler.step()
            # total loss and determine max loss over all batches
            running_loss_train +=loss.item()
            if loss.item() > maxloss:
                maxloss=loss.item()
        # after training, do validation in unseen data without updating gradients
        net.eval()
        plt.figure(3)
        plt.clf()
        plt.plot(Dp_pred.tolist(),Fp_pred.tolist(),'rx',markersize=5)
        plt.ion()
        plt.show()
        # import hiddenlayer as hl
        # hl.build_graph(net.to(device), torch.zeros([60, 60]).to(device))
        
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            X_batch=X_batch.to(device)
            X_pred, _, _, _, _ = net(X_batch)
            loss = criterion(X_pred, X_batch)
            running_loss_val +=loss.item()
            #if loss.item() > maxloss2:
            #    maxloss2=loss.item()
        # scale losses
        running_loss_train=running_loss_train/batch_norm
        running_loss_val=running_loss_val/batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        scheduler.step(running_loss_val)
        if optimizer.param_groups[0]['lr'] < prev_lr:
            net.load_state_dict(final_model)
        prev_lr = optimizer.param_groups[0]['lr']
        # some stuff
        if arg.optim=='sgdr':
            print('Reset scheduler')
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader))
        if arg.run_net == 'loss_con':
            print("Loss: {} of which constrains contributed {}".format(running_loss_train,losstotcon))
        else:
            #print("Loss: {loss}, validation_loss: {val_loss}, max_Loss: {maxloss}, max_validation_loss: {maxloss2}".format(loss=running_loss,val_loss=running_loss_val, maxloss=maxloss, maxloss2=maxloss2))
            print("Loss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,val_loss=running_loss_val, lr=optimizer.param_groups[1]['lr']))
        if arg.run_net == 'loss_con':
            if epoch == 6:
                best = 1e16
        # early stopping criteria
        aaa=1
        if running_loss_val < best:
            print("\n############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss_val
            bestlist.append(best)
            bestepochlist.append(epoch)
            aaa=2
        betterepochs = [bestepochlist[ii] if arg.improve * bestlist[ii] > running_loss_val else None for ii in range(len(bestlist))]
        for ii in range(len(bestlist)-aaa,-1,-1):
            if (arg.improve * bestlist[ii]) > running_loss_val: del bestlist[ii]
        # early stopping if no improvement of at least arg.improve % over the past patience epochs, terminate training
        add_bad_epoch = False
        if not betterepochs:
            add_bad_epoch = True
        betterepochs = [i for i in betterepochs if (i!=None)]
        if not betterepochs:
            add_bad_epoch = True
        elif max(betterepochs) < (epoch-10):
            add_bad_epoch = True
        if add_bad_epoch:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.patience:
                print("\nDone, best val loss: {}".format(best))
                break
        else:
            num_bad_epochs = 0
        if epoch>0:
            plt.figure(1)
            plt.clf()
            plt.plot(b_values, X_pred.data[0])
            plt.plot(b_values, X_batch.data[0])
            plt.yscale("log")
            plt.xlabel('b_val')
            plt.ylabel('signal')
            plt.ion()
            plt.show()
            plt.pause(0.001)
            plt.figure(2)
            plt.clf()
            plt.plot(loss_train)
            plt.plot(loss_val)
            plt.yscale("log")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.ion()
            plt.show()
            plt.pause(0.001)
    print("Done")
    plt.figure(1)
    plt.gcf()
    plt.savefig('C:/Users/ojgurney-champion/Dropbox/Research/DeepLearning/deep_ivim/Output/fig_fit.png')
    plt.figure(2)
    plt.gcf()
    plt.savefig('C:/Users/ojgurney-champion/Dropbox/Research/DeepLearning/deep_ivim/Output/fig_train.png')
    # Restore best model
    net.load_state_dict(final_model)
    return net

def infer_IVIM(data,bvalues,net):
    # this takes a trained network and predicts parameters from it
    # data --> input data to predict parameters from
    # bvalues --> b-value from data
    # net --> network
    
    # use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #normalise signal and remove signal that is clearly corrupted
    S0 = np.mean(data[:,bvalues == 0],axis=1)
    data = data/S0[:,None]
    sels=np.max(data,axis=1)<1.5
    data2=data[sels]

    #tell net it is used for evaluation
    net.eval()
    # initialise parameters and data
    Dp = np.array([])
    Dt = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    inferloader = utils.DataLoader(torch.from_numpy(data2.astype(np.float32)),
                                    batch_size = 2056,
                                    shuffle = False,
                                    drop_last = False)
    # start infering
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader,position=0, leave=True),0):
            X_batch=X_batch.to(device)
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in thepredicted signal decay.
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
    # here we correct for the data tha initially was removed as it did not have IVIM behaviour, by returning zero estimates
    Dptrue=np.zeros(len(data))
    Dttrue=np.zeros(len(data))
    Fptrue=np.zeros(len(data))
    S0true=np.zeros(len(data))
    Dptrue[sels]=Dp
    Dttrue[sels]=Dt
    Fptrue[sels]=Fp
    S0true[sels]=S0
    return [Dptrue, Dttrue, Fptrue, S0true]

def isnan(x):
    # this program indicates what are NaNs 
    return x != x

def pretrain(b, arg, SNR=15, state=1,sims=100000):
    # this allows pretraining a network to help it iterate to the optimum. Input parameters are:
    # b --> b-values to train with
    # arg are the input arguments, with: 
    # arg.run_net the desired network
    # arg.lr the learning rate
    # arg.optim the optimiser
    # arg.patience the patience
    # batch_size is the size of the batches used for training
    # SNR --> gives the SNR
    # state --> optional parameter indicating the random state for random simulation data
    # sims --> number of simulations
    
    # here we simulate Signsl decay for random f, D and Dp
    [S,f, D, Dp] = sim.sim_signal(SNR, b, sims = 100000, Dmin = 0.4 /1000, Dmax = 3.5 /1000, fmin = 0.05, fmax = 0.8, Dsmin= 0.05, Dsmax=0.2, rician = False, state=state)
    # we will only evaluate in the first 5000 D, f and Dp. For training, we do need a lot fo data
    D = D[:5000]
    Dp = Dp[:5000]
    f = f[:5000]
    # here we slowly guide the deep learning into finding the local minimum, starting by training to 1 data, slowly increasing complexity
    meansig=fits.ivim(b, 0.05, 1e-3, 0.25, 1)
    net=learn_IVIM(np.repeat(np.expand_dims(meansig,0),1000,axis=0),b, arg)
    net=learn_IVIM(np.repeat(S[0:100,],10,axis=0),b,arg, net=net)
    net=learn_IVIM(S[0:1000],b,arg,net=net)
    net=learn_IVIM(S,b,arg,net=net)
    # finally net is saved
    torch.save(net.state_dict(), 'results/pretrained_all{net}.pt'.format(net=arg.run_net))
    # and evaluated
    S=S[:5000]
    paramsNN = infer_IVIM(S, b, net)
    matNN=sim.print_errors(np.squeeze(D), np.squeeze(f), np.squeeze(Dp), paramsNN)
    return net, matNN

def loadnet(nets):
    ## this prgram loads a network; nets indicates which one
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