import torch
import torch.nn as nn
from torchinfo import summary
import itertools

class UQPFIN(nn.Module):
    r"""Parameters
    ----------
    particle_feats : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    """

    def __init__(self,
                 particle_feats=3,
                 interaction_feats=4,
                 num_classes=2,
                 Phi_sizes=(100, 100, 64),
                 F_sizes=(64, 100, 100),
                 n_consts = 60,
                 PhiI_nodes = 128,
                 interaction_mode='sum',
                 use_softmax = False,
                 use_dropout = False,
                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

        super(UQPFIN, self).__init__()

        self.Nx = particle_feats
        self.Np = n_consts
        self.Ni = interaction_feats
        self.Npp = (n_consts * (n_consts - 1)) // 2
        self.Nz = Phi_sizes[-1]
        self.PhiI_nodes = PhiI_nodes
        self.device = device
        self.x_mode = interaction_mode if interaction_mode in ['sum', 'cat'] else 'sum'
        self.use_softmax = use_softmax
        self.use_dropout = use_dropout
        #print(Phi_sizes, F_sizes)
        self.assign_matrices()

        # Phi: per-particle function
        phi_layers = []
        for i in range(len(Phi_sizes)):
            if self.use_dropout:
                phi_layers.append(nn.Sequential(
                    nn.Linear(self.Nx if i == 0 else Phi_sizes[i - 1], Phi_sizes[i]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2) if i != len(Phi_sizes) - 1 else nn.Identity() ))
            else:
                phi_layers.append(nn.Sequential(
                    nn.Linear(self.Nx if i == 0 else Phi_sizes[i - 1], Phi_sizes[i]),
                    nn.ReLU() ))
        self.phi = nn.Sequential(*phi_layers)

        # PhiInt: per-particle inteaction
        phiInt_layers = []
        # phiInt_layers.append(nn.Sequential(nn.Linear(2*self.Nx, self.PhiI_nodes),
        #                                    nn.ReLU())
        # )
        if self.use_dropout:
            phiInt_layers.append(nn.Sequential(nn.Linear(self.Ni, self.PhiI_nodes),
                                               nn.ReLU(),
                                               nn.Dropout(p=0.2) ))
            phiInt_layers.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.PhiI_nodes),
                                               nn.ReLU(),
                                               nn.Dropout(p=0.2) ))
        else:
            phiInt_layers.append(nn.Sequential(nn.Linear(self.Ni, self.PhiI_nodes),
                                               nn.ReLU() ))
            phiInt_layers.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.PhiI_nodes),
                                               nn.ReLU() ))
    
        phiInt_layers.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.Nz),
                                           nn.ReLU() ))
        
        self.phiInt = nn.Sequential(*phiInt_layers)

        phiInt_layers2 = []
        
        if self.use_dropout:
            phiInt_layers2.append(nn.Sequential(nn.Linear(self.Nx + self.Nz, self.PhiI_nodes),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.2) ))
            phiInt_layers2.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.PhiI_nodes),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.2) ))
        else:
            phiInt_layers2.append(nn.Sequential(nn.Linear(self.Nx + self.Nz, self.PhiI_nodes),
                                                nn.ReLU() ))
            phiInt_layers2.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.PhiI_nodes),
                                                nn.ReLU() ))

            
        phiInt_layers2.append(nn.Sequential(nn.Linear(self.PhiI_nodes, self.Nz),
                                            nn.ReLU() ))
        
        self.phiInt2 = nn.Sequential(*phiInt_layers2)

        # F: global functions
        f_layers = []
        for i in range(len(F_sizes)):
            if self.use_dropout:
                f_layers.append(nn.Sequential(
                    nn.Linear(((2*self.Nz  if self.x_mode == 'cat' else self.Nz)) if i == 0 else F_sizes[i - 1], F_sizes[i]),
                    nn.ReLU(),
                    nn.Dropout(p=0.2) ))
            else:
                f_layers.append(nn.Sequential(
                    nn.Linear(((2*self.Nz  if self.x_mode == 'cat' else self.Nz)) if i == 0 else F_sizes[i - 1], F_sizes[i]),
                    nn.ReLU() ))
                
        f_layers.append(nn.Linear(F_sizes[-1], num_classes))
        if not self.use_softmax:
            f_layers.append(nn.ReLU())
        else:
            f_layers.append(nn.Softmax(dim=-1))
        self.fc = nn.Sequential(*f_layers)

    def tmul(self, x, y):  # Takes (I * J * K)(K * L) -> I * J * L
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.reshape(-1, x_shape[2]), y).reshape(-1, x_shape[1], y_shape[1])

    def assign_matrices(self):
        # Creates matrices with shape (Np, Npp)
        self.Rr = torch.zeros(self.Np, self.Npp)
        self.Rs = torch.zeros(self.Np, self.Npp)
        receiver_sender_list = [i for i in itertools.product(range(self.Np), range(self.Np)) if i[0] < i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).to(self.device)
        self.Rs = (self.Rs).to(self.device)

    def get_particle_embeddings(self, features, mask):
        # expected features dim: (Nb, Np, Nx)
        # expected mask dim: (Nb, 1, Np)
        # return particle embeddings with dim: (Nb, Nz, Np) 
        features = torch.flatten(features, start_dim=0, end_dim=1)
        x = self.phi(features)
        x = torch.stack(torch.split(x.permute(1, 0), self.Np, dim=1), 0)
        if mask is not None:
            x = x * mask.bool().float()
        return x
        #x = x.sum(-1)
        #return self.fc(x)


    def get_interaction_features(self, E, augmented_feats, mask):
        # expected interaction_feats (E) dim: (Nb, 2Nx, Npp) => (pt0, eta0, phi0, ... , pt1, eta1, phi1, ...)
        # expected mask dim: (Nb, 1, Np)
        # expected augmented feats: (Nb, 7) => jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst
        # return transformed interaction embeddings with dim: (Nb, Ni, Npp)
        Nx = E.shape[1] // 2
        delta = torch.sqrt((E[:,1,:] - E[:,Nx+1,:])**2 + (E[:,2,:] - E[:,Nx+2,:])**2).view(-1,1,self.Npp)

        kT = torch.minimum(E[:,0,:], E[:,Nx,:]).view(-1,1,self.Npp)*delta
        kT = (kT.reshape(-1,self.Npp) * augmented_feats[:,5].reshape(-1,1)).reshape(-1,1,self.Npp)

        z = torch.minimum(E[:,0,:], E[:,Nx,:]).view(-1,1,self.Npp) / (E[:,0,:] + E[:,Nx,:] + 1e-5).view(-1,1,self.Npp)

        e = (E[:,0,:] * augmented_feats[:,5].reshape(-1,1) * torch.cosh(E[:,1,:] + augmented_feats[:,3].reshape(-1,1))).view(-1,1,self.Npp) + \
            (E[:,Nx,:] * augmented_feats[:,5].reshape(-1,1) * torch.cosh(E[:,Nx+1,:] + augmented_feats[:,3].reshape(-1,1))).view(-1,1,self.Npp)
        
        pz = (E[:,0,:] * augmented_feats[:,5].reshape(-1,1) * torch.sinh(E[:,1,:] + augmented_feats[:,3].reshape(-1,1))).view(-1,1,self.Npp) + \
             (E[:,Nx,:] * augmented_feats[:,5].reshape(-1,1) * torch.sinh(E[:,Nx+1,:] + augmented_feats[:,3].reshape(-1,1))).view(-1,1,self.Npp)

        py = (E[:,0,:] * augmented_feats[:,5].reshape(-1,1) * torch.sin(E[:,2,:] + augmented_feats[:,4].reshape(-1,1))).view(-1,1,self.Npp) + \
             (E[:,Nx,:] * augmented_feats[:,5].reshape(-1,1) * torch.sin(E[:,Nx+2,:] + augmented_feats[:,4].reshape(-1,1))).view(-1,1,self.Npp)

        px = (E[:,0,:] * augmented_feats[:,5].reshape(-1,1) * torch.cos(E[:,2,:] + augmented_feats[:,4].reshape(-1,1))).view(-1,1,self.Npp) + \
             (E[:,Nx,:] * augmented_feats[:,5].reshape(-1,1) * torch.cos(E[:,Nx+2,:] + augmented_feats[:,4].reshape(-1,1))).view(-1,1,self.Npp)

        m2 = torch.abs(e**2 - px**2 - py**2 - pz**2)
        #return torch.cat([delta, kT, z, m2], 1) #(Nb, Ni=4, Npp)
        return torch.cat([torch.log(delta + 1e-5), torch.log(kT + 1e-5), torch.log(z + 1e-5), torch.log(m2 + 1e-5)], 1) #(Nb, Ni=4, Npp)

    def get_interaction_embeddings(self, particle_feats, augmented_feats, mask):
        # expected particle_feats dim: (Nb, Np, Nx)
        # expected mask dim: (Nb, 1, Np)
        # expected augmented feats: (Nb, 7) => jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst
        # return per-particle interaction embeddings with dim: (Nb, Nz, Np)
        particle_feats = torch.transpose(particle_feats, 1, 2).contiguous() # (Nb, Nx, Np)
        intR = self.tmul(particle_feats, self.Rr) # (Nb, Nx, Npp)
        intS = self.tmul(particle_feats, self.Rs) # (Nb, Nx, Npp)
        E = torch.cat([intR, intS], 1) # (Nb, 2Nx, Npp)
        #print(E[:5,:,:5])
        # Get interaction features
        E = self.get_interaction_features(E, augmented_feats, mask) # (Nb, Ni, Npp)
        #print(E.shape)
        #print(E[:5,:,:-5])
        
        # Now applying the Interaction MLP
        E = torch.transpose(E, 1, 2).contiguous() #(Nb, Npp, Ni)
        E = self.phiInt(E.view(-1, self.Ni)) # (Nb*Npp, Nz)
        # print(E.shape)
        E = E.view(-1, self.Npp, self.Nz) # (Nb, Npp, Nz)

        if mask is not None:
            # generating masks for interactions
            mR = self.tmul(mask, self.Rr) # (Nb, 1, Npp)
            mS = self.tmul(mask, self.Rs) # (Nb, 1, Npp)
            imask = torch.transpose(mR * mS, 1, 2).contiguous() # (Nb, Npp, 1)
            E = E * imask # (Nb, Npp, Nz) with non-existent interactions masked
        
        
        # Now returning Interactions to particle level inputs
        E = torch.transpose(E, 1, 2).contiguous() # (Nb, Nz, Npp)
        E = ( self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())  \
            + self.tmul(E, torch.transpose(self.Rs, 0, 1).contiguous()) ) / augmented_feats[:,6].reshape(-1,1,1) # (Nb, Nz, Np)
        

        if mask is not None:
            E = E * mask.bool().float()

        # Now concatenaing inputs with first interaction outputs
        E = torch.cat([particle_feats, E], 1) #(Nb, Nx+Nz, Np)
        E = torch.transpose(E, 1, 2).contiguous() #(Nb, Np, Nx+Nz)
        E = self.phiInt2(E.view(-1, self.Nx + self.Nz)) #(Nb*Np, Nz)
        E = E.view(-1, self.Np, self.Nz) #(Nb, Np, Nz)
        E = torch.transpose(E, 1, 2).contiguous() #(Nb, Nz, Np)
        
        if mask is not None:
            E = E * mask.bool().float()
        
        return E

    def forward(self, features, aug, mask):
        # expected features dim: (Nb, Np, Nx)
        # expected mask dim: (Nb, 1, Np)
        # # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        # #features = features.permute(0,2,1)
        # features = torch.flatten(features, start_dim=0, end_dim=1)
        # x = self.phi(features)
        # x = torch.stack(torch.split(x.permute(1, 0), self.Np , dim=1), 0)
        # if mask is not None:
        #     x = x * mask.bool().float()
        particle_embeddings = self.get_particle_embeddings(features, mask)
        interaction_embeddings = self.get_interaction_embeddings(features, aug, mask)
        if self.x_mode == 'sum':
            x = particle_embeddings.sum(-1) + interaction_embeddings.sum(-1)
        else:
            x = torch.cat([particle_embeddings, interaction_embeddings], 1)
            x = x.sum(-1)
        return self.fc(x)
        

if __name__ == "__main__":
    features = 3
    n_consts = 60
    x_mode = 'cat'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UQPFIN(particle_feats = features,
                   n_consts = n_consts,
                   interaction_mode = x_mode,
                   Phi_sizes=(100, 100, 64),
                   F_sizes=(64, 64, 64),
                   device = device).to(device)
    x = torch.rand(2, n_consts, features).to(device)
    mask = torch.cat([torch.ones(2, 1, (n_consts // 2)), torch.zeros(2, 1, n_consts - (n_consts // 2))], 2).float().to(device)
    aug = torch.ones(2,7).float().to(device)
    with torch.no_grad():
        p_emb = model.get_particle_embeddings(x, mask)
        print("Shape of Particle embedding: ", p_emb.shape)
        p_emb = model.get_interaction_embeddings(x, aug, mask)
        print("Shape of Interaction embedding: ", p_emb.shape)
        y = model(x, aug, mask)
        print(y)
    summary(model, ((1, n_consts, features), (1,7), (1, 1, n_consts)))