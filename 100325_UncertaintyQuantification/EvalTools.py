import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import argparse, os, json, sys
import h5py
sys.path.append("../model")
from PFINDataset import PFINDataset, JetClassData
from UQPFIN import UQPFIN as Model
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt

def getprobs(outs):
    alphas = outs + 1
    S = torch.sum(alphas, 1).reshape(-1,1)
    return alphas / S

def index_logic(logic, v1, v2, a):
    if logic == 'AND':
        keep = (a > min(v1, v2)) & (a < max(v1, v2))
    else:
        keep = (a < min(v1, v2)) | (a > max(v1, v2))
    return keep


class ModelEvaluator:
    def __init__(self, model_path, evalMode = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dict_path = model_path.replace("_best","").replace("_best","").replace("trained_models/", "trained_model_dicts/") + ".json"
        self.model_dict = json.load(open(self.model_dict_path))
        self.phi_nodes = list(map(int, self.model_dict["phi_nodes"].strip().split(',')))
        self.f_nodes = list(map(int, self.model_dict["f_nodes"].strip().split(',')))
        self.n_phiI = int(self.model_dict['n_phiI'])
        self.label = self.model_dict['label']
        self.data_type = self.model_dict['data_type']
        self.massrange = self.model_dict['massrange']
        self.etarange = self.model_dict['etarange']
        self.ptrange = self.model_dict['ptrange']
        self.x_mode = self.model_dict['x_mode']
        self.skip_labels = self.model_dict["skiplabels"]
        try:
            self.use_softmax = self.model_dict["use_softmax"]
        except:
            self.use_softmax = False
        try:
            self.use_dropout = self.model_dict["use_dropout"]
        except:
            self.use_dropout = False
        self.m_logic, (self.m1, self.m2) = self.massrange.strip().split(':')[0], list(map(float, self.massrange.strip().split(':')[1].split(',')))
        self.pt_logic,(self.pt1, self.pt2) = self.ptrange.strip().split(':')[0], list(map(float, self.ptrange.strip().split(':')[1].split(',')))
        self.eta_logic, (self.eta1, self.eta2) = self.etarange.strip().split(':')[0], list(map(float, self.etarange.strip().split(':')[1].split(',')))
        
        if self.data_type == 'topdata':
            features = 3
            Np = 60
            self.num_classes = 2
            self.skip_labels = []
            self.label_indices = [0,1]
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            features = 3
            Np = 30
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 5 - len(self.skip_labels)
            self.label_indices = [i for i in range(5) if i not in self.skip_labels]
            self.data_path = "../datasets/jetnet/test.h5"
        elif self.data_type == 'jetclass':
            features = 11
            Np = 60
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 10 - len(self.skip_labels)
            self.label_indices = [i for i in range(10) if i not in self.skip_labels]
            self.data_path = glob.glob(os.path.join("../datasets/jetclass", "test_*.h5"))
            
        elif self.data_type == 'JNqgmerged':
            features = 3
            Np = 30
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 4 - len(self.skip_labels)
            self.label_indices = [i for i in range(4) if i not in self.skip_labels]
            self.data_path = "../datasets/JNqgmerged/test.h5"
            
        self.model = Model(particle_feats = features,
                           n_consts = Np,
                           num_classes = self.num_classes,
                           device = self.device ,
                           PhiI_nodes = self.n_phiI,
                           interaction_mode = self.x_mode,
                           use_softmax = self.use_softmax,
                           use_dropout = self.use_dropout,
                           Phi_sizes = self.phi_nodes,
                           F_sizes   = self.f_nodes).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
    
        self.model.load_state_dict(new_state_dict)
        
        if evalMode:
            self.model.eval()
        else:
            self.model.train()
        

        
    def index_groomer(self, a, y):
        keep_masses = index_logic(self.m_logic,   self.m1,   self.m2,   a[:, 1])
        keep_pts    = index_logic(self.pt_logic,  self.pt1,  self.pt2,  a[:, 2])
        keep_etas   = index_logic(self.eta_logic, self.eta1, self.eta2, a[:, 3])
        
        keep_labels = torch.tensor(np.isin(np.argmax(y.cpu().numpy(), 1), self.skip_labels, invert=True)).bool()
        
        return (keep_masses & keep_pts & keep_etas & keep_labels).bool()
    
    def evaluate(self, data_loader = None, latent = False, aug = False, batchmode = False):
        if not data_loader:
            if self.data_type == 'topdata' or self.data_type == 'jetnet' or self.data_type == "JNqgmerged":
                test_set = PFINDataset(self.data_path)
                testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            elif self.data_type == 'jetclass':
                test_set = JetClassData(batch_size = 512)
                test_set.set_file_names(file_names = self.data_path)
                testloader = test_set.generate_data()
        else:
            testloader = data_loader
        labels = []
        preds = []
        maxprobs = []
        probs = []
        oods = []
        sums = []
        uncs = []
        latents = []
        aug_data = []
        with torch.no_grad():
            for x,m,a,y in tqdm(testloader, disable=batchmode):
                x = x.cuda()
                m = m.cuda()
                a = a.cuda()

                pred = self.model(x, a, m).cpu()
                idx2keep = self.index_groomer(a.cpu(), y).cpu().numpy()
                if not self.use_softmax:
                    model_prob = getprobs(pred)
                else:
                    model_prob = pred.clone()
                all_prob = torch.zeros_like(y)
                all_prob[:, self.label_indices] = model_prob
                y = y.cpu().numpy()
                all_prob = all_prob.cpu().numpy()
                
                if latent:
                    particle_embeddings = self.model.get_particle_embeddings(x, m)
                    interaction_embeddings = self.model.get_interaction_embeddings(x, a, m)
                    if self.model.x_mode == 'sum':
                        latent_embeddings = particle_embeddings.sum(-1) + interaction_embeddings.sum(-1)
                    else:
                        latent_embeddings = torch.cat([particle_embeddings, interaction_embeddings], 1).sum(-1)
                    latents.append(latent_embeddings.cpu().numpy())
                if aug:
                    aug_data.append(a.cpu().numpy())
                labels.append(np.argmax(y, 1))
                preds.append(np.argmax(all_prob, 1))
                maxprobs.append(all_prob.max(axis=-1))
                probs.append(all_prob)
                oods.append(~idx2keep)
                if not self.use_softmax:
                    sums.append((pred+1).sum(-1).cpu().numpy())

        labels = np.concatenate(labels, axis = None)
        preds = np.concatenate(preds, axis = None)
        oods = np.concatenate(oods, axis = None)
        maxprobs = np.concatenate(maxprobs, axis = None)
        probs = np.concatenate(probs, axis = 0)
        if not self.use_softmax:
            sums = np.concatenate(sums, axis = None)
            uncs = len(self.label_indices)*1.0 / sums
        if latent:
            latents = np.concatenate(latents, axis=0)
        if aug:
            aug_data = np.concatenate(aug_data, axis=0)

        
        if not data_loader:
            del test_set, testloader
            
        if not aug and not latent:
            return labels, preds, maxprobs, probs, sums, oods, uncs
        elif aug and not latent:
            return labels, preds, maxprobs, probs, sums, oods, uncs, aug_data
        elif not aug and latent:
            return labels, preds, maxprobs, probs, sums, oods, uncs, latents
        else:
            return labels, preds, maxprobs, probs, sums, oods, uncs, aug_data, latents
    

class EnsembleEvaluator:
    def __init__(self, model_paths, data_type = "jetnet"):
        self.model_paths = model_paths
        self.data_type = data_type
        if self.data_type == 'topdata':
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            self.data_path = "../datasets/jetnet/test.h5"
        elif self.data_type == 'jetclass':
            self.data_path = glob.glob(os.path.join("../datasets/jetclass", "test_*.h5"))
        elif self.data_type == "JNqgmerged":
            self.data_path = "../datasets/JNqgmerged/test.h5"
            
    def evaluate(self, test_set = None, aug = False, batchmode = False):
        if not test_set:
            if self.data_type == 'topdata' or self.data_type == 'jetnet' or self.data_type == "JNqgmerged":
                test_set = PFINDataset(self.data_path)
                testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            elif self.data_type == 'jetclass':
                test_set = JetClassData(batch_size = 512)
                test_set.set_file_names(file_names = self.data_path)
            delete_test_set = True
        elif type(test_set) == PFINDataset:
            testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            delete_test_set = False
        else:
            delete_test_set = False
            
        preds = []
        maxprobs = []
        probs = []
        sums = []
        for ii, model_path in enumerate(self.model_paths):
            model_evaluator = ModelEvaluator(model_path, evalMode = True)
            if type(test_set) == JetClassData:
                testloader = test_set.generate_data()
            if ii == 0 and aug:
                labels, _, _, this_probs, _, oods, _, aug_data =  model_evaluator.evaluate(testloader, aug = True, batchmode = batchmode)
            elif ii == 0:
                labels, _, _, this_probs, _, oods, _ =  model_evaluator.evaluate(testloader, batchmode = batchmode)
            else:
                _, _, _, this_probs, _, _, _ =  model_evaluator.evaluate(testloader, batchmode = batchmode)
            probs.append(this_probs[None,:,:])
            torch.cuda.empty_cache()
            del model_evaluator
                
                
        probs = np.concatenate(probs, axis = 0)
        uncs = np.std(probs, axis = 0).max(axis = 1)
        probs = np.mean(probs, axis = 0)
        preds = np.argmax(probs, 1)
        maxprobs = probs.max(axis = 1)
                
        del testloader
        if delete_test_set:
            del test_set
            
        if aug:
            return labels, preds, maxprobs, probs, sums, oods, uncs, aug_data
        else:
            return labels, preds, maxprobs, probs, sums, oods, uncs

class MCDOEvaluator:
    def __init__(self, model_path, data_type = "jetnet"):
        self.model_path = model_path
        self.data_type = data_type
        if self.data_type == 'topdata':
            self.data_path = "../datasets/topdata/test.h5"
        elif self.data_type == 'jetnet':
            self.data_path = "../datasets/jetnet/test.h5"
        elif self.data_type == 'jetclass':
            self.data_path = glob.glob(os.path.join("../datasets/jetclass", "test_*.h5"))
        elif self.data_type == "JNqgmerged":
            self.data_path = "../datasets/JNqgmerged/test.h5"
            
    def evaluate(self, test_set = None, aug = False, batchmode = False):
        if not test_set:
            if self.data_type == 'topdata' or self.data_type == 'jetnet' or self.data_type == "JNqgmerged":
                test_set = PFINDataset(self.data_path)
                testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            elif self.data_type == 'jetclass':
                test_set = JetClassData(batch_size = 512)
                test_set.set_file_names(file_names = self.data_path)
            delete_test_set = True
        elif type(test_set) == PFINDataset:
            testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            delete_test_set = False
        else:
            delete_test_set = False
            
        preds = []
        maxprobs = []
        probs = []
        sums = []
        model_evaluator = ModelEvaluator(self.model_path, evalMode = False)
        for ii in range(10):
            if type(test_set) == JetClassData:
                testloader = test_set.generate_data()
            if ii == 0 and aug:
                labels, _, _, this_probs, _, oods, _, aug_data =  model_evaluator.evaluate(testloader, aug = True, batchmode = batchmode)
            elif ii == 0:
                labels, _, _, this_probs, _, oods, _ =  model_evaluator.evaluate(testloader, batchmode = batchmode)
            else:
                _, _, _, this_probs, _, _, _ =  model_evaluator.evaluate(testloader, batchmode = batchmode)
            probs.append(this_probs[None,:,:])
            torch.cuda.empty_cache()
        del model_evaluator
                
                
        probs = np.concatenate(probs, axis = 0)
        uncs = np.std(probs, axis = 0).max(axis = 1)
        probs = np.mean(probs, axis = 0)
        preds = np.argmax(probs, 1)
        maxprobs = probs.max(axis = 1)
            
        del testloader
        if delete_test_set:
            del test_set
            
        if aug:
            return labels, preds, maxprobs, probs, sums, oods, uncs, aug_data
        else:
            return labels, preds, maxprobs, probs, sums, oods, uncs

    
class PlotterTools:
    def __init__(self, model_results, tag):
        self.uncs = model_results['uncs']
        self.preds = model_results['preds']
        self.labels = model_results['labels']
        self.oods = model_results['oods']
        self.probs = model_results['probs']
        self.sums = model_results['sums']
        self.tag = tag
        ul, ur = self.uncs.min(), self.uncs.max()
        du = (ur - ul)/100
        self.urange = np.arange(ul+du, ur-du, du)
        
    def ID_Unc(self, ax):
        # todo
        
        IDAccs = []
        
        for u in self.urange:
            pred_id_indices = (self.uncs >= u) & (~self.oods)
            idacc = (self.labels[pred_id_indices] != self.preds[pred_id_indices]).sum()/(~self.oods).sum()
            IDAccs.append(idacc)
            
        ax.plot(self.urange, IDAccs, label=self.tag)
        
    def ODR_IDAcc(self, ax):
        # plots OOD Detection Rate vs ID Accuracy for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD
        # IDAcc = Correct_ID_tag / Total_ID
        
        ODRs = []
        IDAccs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = (self.uncs < u) & (~self.oods)
            idacc = (self.labels[pred_id_indices] == self.preds[pred_id_indices]).sum()/(~self.oods).sum()
            IDAccs.append(idacc)
            
        ax.plot(IDAccs, ODRs, label=self.tag)
        
    def ODR_Acc(self, ax):
        # plots OOD Detection Rate vs Accuracy for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD
        # Acc = Correct_OOD or Correctly_tagged_ID / N
        
        ODRs = []
        Accs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = self.uncs < u
            acc = ((self.labels[pred_id_indices] == self.preds[pred_id_indices]).sum() + (self.oods & (self.uncs >= u)).sum())/len(self.labels)
            Accs.append(acc)
            
        ax.plot(Accs, ODRs, label=self.tag)
        
    def OCR_ODR(self, ax):
        # plots OOD Detection Rate vs OOD Conf Rate for differenct unc thresholds
        # ODR = OOD_Positive / Total_OOD (like recall or True-Positive-Rate)
        # OCR = OOD_Positive / Total_Pred_OOD (like Precision)
        
        ODRs = []
        OCRs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            ocr = (self.oods & (self.uncs >= u)).sum()/(self.uncs >= u).sum()
            OCRs.append(ocr)
            
        ax.plot(ODRs, OCRs, label=self.tag)
        
    def ODR_IMR(self, ax):
        # plots OOD Detection Rate vs ID Mistag Rate for differenct unc thresholds (equivalent to ROC)
        # ODR = OOD_Positive / Total_OOD
        # IMR = ID_Positive / Total_ID
        
        ODRs = []
        IMRs = []
        
        for u in self.urange:
            odr = (self.oods & (self.uncs >= u)).sum()/self.oods.sum()
            ODRs.append(odr)
            pred_id_indices = (self.uncs >= u) & (~self.oods)
            imr = pred_id_indices.sum()/(~self.oods).sum()
            IMRs.append(imr)
            
        ax.plot(IMRs, ODRs, label=self.tag)
        
        
    def RAR_RER(self, ax):
        #plots Remaining Accuracy Rate v Remaining Error Rate [https://ceur-ws.org/Vol-2640/paper_18.pdf]
        # RAR = Correct and Confident / N = [(large unc and OOD) or (small uncertainty and correct ID)] / N
        # RER = Incorrect and Confident / N = [(large unc and ID) or (small unc and OOD) ] / N
        
        RARs = []
        RERs = []
        
        for u in self.urange:
            CC_ood = ((self.uncs >= u) & self.oods).sum()
            CC_id_indices = (self.uncs < u) & (~self.oods)
            CC_id = (self.labels[CC_id_indices] == self.preds[CC_id_indices]).sum()
            rar = (CC_ood + CC_id)/len(self.labels)
            
            IC_id = ((self.uncs >= u) & ~self.oods).sum()
            IC_ood = ((self.uncs < u) & self.oods).sum()
            rer = (IC_ood + IC_id)/len(self.labels)
            
            RARs.append(rar)
            RERs.append(rer)
            
        ax.plot(RERs, RARs, label=self.tag)
    

    def UNC_ENTROPY(self, ax):
        # plots Uncertainty vs Entropy 
        def reject_outliers(data, m=2):
            return data[abs(data - np.mean(data)) < m * np.std(data)]
    
        ENTs = []
        p = self.probs.copy()
        p = np.where(p == 0, 1, p)
        entropy = -np.sum(p * np.log(p), axis=1)
            
        y = self.uncs
        
        ul, ur = entropy.min(), entropy.max()
        bins = np.linspace(ul, ur, 200)
        digitized = np.digitize(entropy, bins)
        
        bin_means = np.array([y[digitized == i].mean() for i in range(1, len(bins))])
        bin_stds = np.array([y[digitized == i].std() for i in range(1, len(bins))])
        bin_centers = np.array([bins[i-1:i+1].mean() for i in range(1, len(bins))])       
                        
        ax.plot(bin_centers, bin_means, label=self.tag, alpha=0.5, linewidth=3)
        ax.fill_between(bin_centers, bin_means-bin_stds, bin_means+bin_stds, alpha=0.5)
        
        
    def CDF_ENTROPY(self, ax):
        # plots CDF of Entropy 
        
        ENTs = []
        p = self.probs.copy()
        p = np.where(p == 0, 1, p)
        entropy = -np.sum(p * np.log(p), axis=1)

        ul, ur = entropy.min(), entropy.max()
        erange = np.linspace(ul, ur, 200)
                
        for i in range(len(erange)):
            
            prob = np.sum(entropy <= erange[i]) / len(entropy)
            ENTs.append(prob)
        
                        
        ax.plot(erange, ENTs, label=self.tag)
        
   
    def UNC_FOURPLOT(self, ax):
        tp_rate = []
        fp_rate = []
        fn_rate = []
        tn_rate = []
        
        for i in range(len(self.urange)):
            f = self.uncs >= self.urange[i]
            tp = (self.oods & f).sum()/self.oods.sum()
            fp = (~self.oods & f).sum()/(~self.oods).sum()
            fn = (self.oods & ~f).sum()/self.oods.sum()
            tn = (~self.oods & ~f).sum()/(~self.oods).sum()

            tp_rate.append(tp)
            fp_rate.append(fp)
            fn_rate.append(fn)
            tn_rate.append(tn)

        ax[0, 0].plot(self.urange, tp_rate, label=self.tag)
        ax[0, 1].plot(self.urange, fp_rate, label=self.tag)
        ax[1, 0].plot(self.urange, fn_rate, label=self.tag)
        ax[1, 1].plot(self.urange, tn_rate, label=self.tag)

        
        
def getTrackID(Np, mask_index):
    return torch.sum(torch.arange((Np - 1), (Np - 1) + mask_index[0]*(-1), -1)).item() + \
                               mask_index[1] - mask_index[0] - 1

def removeOutliers(x, lq = 0.25, rq = 0.75, outlierConstant=5):
    quantiles = np.nanquantile(x, (lq, rq),axis=0)
    IQR = (quantiles[1]-quantiles[0])*outlierConstant
    result = np.where((x>=quantiles[0]-IQR)&(x<=quantiles[1]+IQR), x, 0)
    
    return result

def interaction_features(model, particle_feats, augmented_feats, mask):
    # expected particle_feats dim: (Nb, Np, Nx)
    # expected mask dim: (Nb, 1, Np)
    # expected augmented feats: (Nb, 7) => jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst
    # return per-particle interaction embeddings with dim: (Nb, Nz, Np)
    particle_feats = torch.transpose(particle_feats, 1, 2).contiguous() # (Nb, Nx, Np)
    intR = model.tmul(particle_feats, model.Rr) # (Nb, Nx, Npp)
    intS = model.tmul(particle_feats, model.Rs) # (Nb, Nx, Npp)
    E = torch.cat([intR, intS], 1) # (Nb, 2Nx, Npp)
    #print(E[:5,:,:5])
    # Get interaction features
    E = model.get_interaction_features(E, augmented_feats, mask) # (Nb, Ni, Npp)
    #print(E.shape)
    #print(E[:5,:,:-5])
    return E # Returns interaction features

def interaction_embeddings_calc(model, E, particle_feats, augmented_feats, mask):
    # Now applying the Interaction MLP
    particle_feats = torch.transpose(particle_feats, 1, 2).contiguous()
    E = torch.transpose(E, 1, 2).contiguous() #(Nb, Npp, Ni)
    E = model.phiInt(E.view(-1, model.Ni)) # (Nb*Npp, Nz)
    # print(E.shape)
    E = E.view(-1, model.Npp, model.Nz) # (Nb, Npp, Nz)

    if mask is not None:
        # generating masks for interactions
        mR = model.tmul(mask, model.Rr) # (Nb, 1, Npp)
        mS = model.tmul(mask, model.Rs) # (Nb, 1, Npp)
        imask = torch.transpose(mR * mS, 1, 2).contiguous() # (Nb, Npp, 1)
        E = E * imask # (Nb, Npp, Nz) with non-existent interactions masked


    # Now returning Interactions to particle level inputs
    E = torch.transpose(E, 1, 2).contiguous() # (Nb, Nz, Npp)
    E = ( model.tmul(E, torch.transpose(model.Rr, 0, 1).contiguous())  \
        + model.tmul(E, torch.transpose(model.Rs, 0, 1).contiguous()) ) / augmented_feats[:,6].reshape(-1,1,1) # (Nb, Nz, Np)


    if mask is not None:
        E = E * mask.bool().float()

    # Now concatenaing inputs with first interaction outputs
    E = torch.cat([particle_feats, E], 1) #(Nb, Nx+Nz, Np)
    E = torch.transpose(E, 1, 2).contiguous() #(Nb, Np, Nx+Nz)
    E = model.phiInt2(E.view(-1, model.Nx + model.Nz)) #(Nb*Np, Nz)
    E = E.view(-1, model.Np, model.Nz) #(Nb, Np, Nz)
    E = torch.transpose(E, 1, 2).contiguous() #(Nb, Nz, Np)

    if mask is not None:
        E = E * mask.bool().float()

    return E
        
class PairwiseEvaluator:
    def __init__(self, model_path, evalMode = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dict_path = model_path.replace("_best","").replace("_best","").replace("trained_models/", "trained_model_dicts/") + ".json"
        self.model_dict = json.load(open(self.model_dict_path))
        self.phi_nodes = list(map(int, self.model_dict["phi_nodes"].strip().split(',')))
        self.f_nodes = list(map(int, self.model_dict["f_nodes"].strip().split(',')))
        self.n_phiI = int(self.model_dict['n_phiI'])
        self.label = self.model_dict['label']
        self.data_type = self.model_dict['data_type']
        self.massrange = self.model_dict['massrange']
        self.etarange = self.model_dict['etarange']
        self.ptrange = self.model_dict['ptrange']
        self.x_mode = self.model_dict['x_mode']
        self.skip_labels = self.model_dict["skiplabels"]
        try:
            self.use_softmax = self.model_dict["use_softmax"]
        except:
            self.use_softmax = False
        try:
            self.use_dropout = self.model_dict["use_dropout"]
        except:
            self.use_dropout = False
        self.m_logic, (self.m1, self.m2) = self.massrange.strip().split(':')[0], list(map(float, self.massrange.strip().split(':')[1].split(',')))
        self.pt_logic,(self.pt1, self.pt2) = self.ptrange.strip().split(':')[0], list(map(float, self.ptrange.strip().split(':')[1].split(',')))
        self.eta_logic, (self.eta1, self.eta2) = self.etarange.strip().split(':')[0], list(map(float, self.etarange.strip().split(':')[1].split(',')))
        
        if self.data_type == 'topdata':
            features = 3
            Np = 60
            self.num_classes = 2
            self.skip_labels = []
            self.label_indices = [0,1]
            self.data_path = "../datasets/topdata/test.h5"
        if self.data_type == 'jetnet':
            features = 3
            Np = 30
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 5 - len(self.skip_labels)
            self.label_indices = [i for i in range(5) if i not in self.skip_labels]
            self.data_path = "../datasets/jetnet/test.h5"
        elif self.data_type == 'jetclass':
            features = 11
            Np = 60
            if self.skip_labels:
                self.skip_labels = list(map(int, self.skip_labels.strip().split(',')))
            else:
                self.skip_labels = []
            self.num_classes = 10 - len(self.skip_labels)
            self.label_indices = [i for i in range(10) if i not in self.skip_labels]
            self.data_path = glob.glob(os.path.join("../jetclass", "test_*.h5"))
        
        self.model = Model(particle_feats = features,
                           n_consts = Np,
                           num_classes = self.num_classes,
                           device = self.device ,
                           PhiI_nodes = self.n_phiI,
                           interaction_mode = self.x_mode,
                           use_softmax = self.use_softmax,
                           use_dropout = self.use_dropout,
                           Phi_sizes = self.phi_nodes,
                           F_sizes   = self.f_nodes).to(self.device)
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
    
        self.model.load_state_dict(new_state_dict)
        
        if evalMode:
            self.model.eval()
        else:
            self.model.train()
        

        
    def index_groomer(self, a, y):
        keep_masses = index_logic(self.m_logic,   self.m1,   self.m2,   a[:, 1])
        keep_pts    = index_logic(self.pt_logic,  self.pt1,  self.pt2,  a[:, 2])
        keep_etas   = index_logic(self.eta_logic, self.eta1, self.eta2, a[:, 3])

        keep_labels = torch.tensor(np.isin(np.argmax(y.cpu().numpy(), 1), self.skip_labels, invert=True)).bool()

        return (keep_masses & keep_pts & keep_etas & keep_labels).bool()
    
    def evaluate(self, data_loader = None, mask_index = None):
        if not data_loader:
            if self.data_type == 'topdata' or self.data_type == 'jetnet' or self.data_type == "JNqgmerged":
                test_set = PFINDataset(self.data_path)
                testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=2, pin_memory=True, persistent_workers=True)
            elif self.data_type == 'jetclass':
                test_set = JetClassData(batch_size = 512)
                test_set.set_file_names(file_names = self.data_path)
                testloader = test_set.generate_data()
        else:
            testloader = data_loader
        masked_track = None
        
        F_layers = [layer for layer in self.model.fc[:-1]]
        modelFC_ps = nn.Sequential(*F_layers)
        if self.use_softmax:
            activation = nn.Softmax(dim=1)
        else:
            activation = nn.ReLU()
                
        labels = []
        preprobs = []
        probs = []
        sums = []
        uncs = []
        latents = []
        intfeat = []
        with torch.no_grad():
            for idx, (x,m,a,y) in enumerate(testloader):
                x = x.cuda()
                m = m.cuda()
                a = a.cuda()
                
                particle_embeddings = self.model.get_particle_embeddings(x, m)
                E = interaction_features(self.model, x, a, m)
                if mask_index != None and len(mask_index) == 2 and mask_index[0] < mask_index[1]:
                    masked_track = getTrackID(self.model.Np, mask_index)
                    m[:, :, mask_index] = 0 
                interaction_embeddings = interaction_embeddings_calc(self.model, E, x, a, m)
                            
                if self.model.x_mode == 'sum':
                    latent_embeddings = particle_embeddings.sum(-1) + interaction_embeddings.sum(-1)
                else:
                    latent_embeddings = torch.cat([particle_embeddings, interaction_embeddings], 1).sum(-1)
                    
                preprob = modelFC_ps(latent_embeddings)
                if self.use_softmax:
                    prob = activation(preprob)
                else:
                    preprob = activation(preprob)
                    prob = getprobs(preprob)

                latents.append(latent_embeddings.cpu().numpy())
                labels.append(np.argmax(y, 1))
                intfeat.append(E.cpu().numpy())
                preprobs.append(preprob.cpu().numpy())
                probs.append(prob.cpu().numpy())
                
                if not self.use_softmax:
                    sums.append((preprob+1).sum(-1).cpu().numpy())
                if idx == 100:
                    break
                
        labels = np.concatenate(labels, axis = 0)
        preprobs = np.concatenate(preprobs, axis = 0)
        latents = np.concatenate(latents, axis=0)
        probs = np.concatenate(probs, axis = 0)
        intfeat = np.concatenate(intfeat, axis = 0)
        if not data_loader:
            del test_set, testloader
            
        if not self.use_softmax:
            sums = np.concatenate(sums, axis = None)
            uncs = len(self.label_indices)*1.0 / sums
        
                    
        return latents, labels, intfeat, preprobs, probs, masked_track, uncs        
        
        
def uncertainty_plot(uncs, oods, maxprobs, labels, preds, key, l_max, fsize, tsize, asize, asize2):
    fig, axes = plt.subplots(2, 4, figsize=(8*5, 8*2), sharex = False, sharey = False)
    
    axes[0, 0].hist(uncs[~oods],
                    bins=np.arange(0.,1.01,0.04), 
                    label=key + '(~oods)', 
                    alpha = 0.7, 
                    histtype = 'step', linewidth = 3)
    axes[0, 0].set_xlabel("Uncertainty", fontsize=fsize)
    axes[0, 0].tick_params(axis='both', labelsize=tsize)
    axes[1, 0].hist(uncs[oods],
                    bins=np.arange(0.,1.01,0.04), 
                    label=key + '(oods)', 
                    alpha = 0.7, 
                    histtype = 'step', linewidth = 3)
    axes[1, 0].set_xlabel("Uncertainty", fontsize=fsize)
    axes[1, 0].tick_params(axis='both', labelsize=tsize)
    
    
    h = axes[0, 1].hist2d(maxprobs[~oods], uncs[~oods], 
                          cmap = 'winter',
                          bins = [np.arange(0.,1.01,0.04), np.arange(0.,1.01,0.04)])
    cbar = fig.colorbar(h[3], ax=axes[0, 1])
    axes[0, 1].set_xlabel("Max. Prob.", fontsize=fsize)
    axes[0, 1].set_ylabel("Uncertainty", fontsize=fsize)
    axes[0, 1].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    
    h = axes[1, 1].hist2d(maxprobs[oods], uncs[oods], 
                          cmap = 'winter',
                          bins = [np.arange(0.,1.01,0.04), np.arange(0.,1.01,0.04)])
    cbar = fig.colorbar(h[3], ax=axes[1, 1])
    axes[1, 1].set_xlabel("Max. Prob.", fontsize=fsize)
    axes[1, 1].set_ylabel("Uncertainty", fontsize=fsize)
    axes[1, 1].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    
    
    h = axes[0, 2].hist2d(labels[~oods], preds[~oods], 
                          cmap = 'winter',
                          bins = [np.arange(0.,l_max,1), np.arange(0.,l_max,1)])
    
    for i in range(len(h[2])-1):
        for j in range(len(h[1])-1):
            axes[0,2].text(h[1][j]+0.5,h[2][i]+0.5, int(h[0].T[i,j]), 
                           color="w", ha="center", va="center", fontweight="bold", fontsize=asize)

    cbar = fig.colorbar(h[3], ax=axes[0, 2])
    axes[0, 2].set_xlabel("Labels", fontsize=fsize)
    axes[0, 2].set_ylabel("Preds", fontsize=fsize)
    axes[0, 2].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    
    h = axes[1, 2].hist2d(labels[oods], preds[oods], 
                          cmap = 'winter',
                          bins = [np.arange(0.,l_max,1), np.arange(0.,l_max,1)])
    for i in range(len(h[2])-1):
        for j in range(len(h[1])-1):
            axes[1,2].text(h[1][j]+0.5,h[2][i]+0.5, int(h[0].T[i,j]), 
                           color="w", ha="center", va="center", fontweight="bold", fontsize=asize)
    cbar = fig.colorbar(h[3], ax=axes[1, 2])
    axes[1, 2].set_xlabel("Labels", fontsize=fsize)
    axes[1, 2].set_ylabel("Preds", fontsize=fsize)
    axes[1, 2].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    
    
    h = axes[0, 3].hist2d(labels[~oods], uncs[~oods] + preds[~oods] -1.e-4, 
                          cmap = 'winter',
                          bins = [np.arange(0.,l_max,1), np.arange(0.,l_max,0.2)])
    cbar = fig.colorbar(h[3], ax=axes[0, 3])
    axes[0, 3].set_xlabel("Labels", fontsize=fsize)
    axes[0, 3].set_ylabel("Pred. Label + Unc.", fontsize=fsize)
    axes[0, 3].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    for i in range(len(h[2])-1):
        for j in range(len(h[1])-1):
            axes[0,3].text(h[1][j]+0.5,h[2][i]+0.05, int(h[0].T[i,j]), 
                           color="w", ha="center", va="center", fontweight="bold", fontsize=asize2)
    for jj in np.arange(1,l_max,1):
        axes[0, 3].axhline(jj)
    
    h = axes[1, 3].hist2d(labels[oods], uncs[oods] + preds[oods] -1.e-4, 
                          cmap = 'winter',
                          bins = [np.arange(0.,l_max,1), np.arange(0.,l_max,0.2)])
    cbar = fig.colorbar(h[3], ax=axes[1, 3])
    axes[1, 3].set_xlabel("Labels", fontsize=fsize)
    axes[1, 3].set_ylabel("Pred. Label + Unc.", fontsize=fsize)
    axes[1, 3].tick_params(axis='both', labelsize=tsize)
    cbar.ax.tick_params(axis='y', labelsize=tsize)
    for i in range(len(h[2])-1):
        for j in range(len(h[1])-1):
            axes[1,3].text(h[1][j]+0.5,h[2][i]+0.05, int(h[0].T[i,j]), 
                           color="w", ha="center", va="center", fontweight="bold", fontsize=asize2)
            
    for jj in np.arange(1,l_max,1):
        axes[1, 3].axhline(jj)
            