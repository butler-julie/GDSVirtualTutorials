import pandas as pd
import math 
import numpy as np
import glob
import sklearn
import sys, os
import h5py
from sklearn.metrics import roc_curve


def get_pt_eta_phi_v(px, py, pz):
    '''Provides pt, eta, and phi given px, py, pz'''
    # Init variables
    pt = np.zeros(len(px))
    pt = np.sqrt(np.power(px,2) + np.power(py,2))
    phi = np.zeros(len(px))
    eta = np.zeros(len(px))
    theta = np.zeros(len(px))
    x = np.where((px!=0) | (py!=0) | (pz!=0)) # locate where px,py,pz are all 0 
    theta[x] = np.arctan2(pt[x],pz[x]) 
    cos_theta = np.cos(theta)
    y = np.where(np.power(cos_theta,2) < 1)
    eta[y] = -0.5*np.log((1 - cos_theta[y]) / (1 + cos_theta[y]))
    z = np.where((px !=0)|(py != 0))
    phi[z] = np.arctan2(py[z],px[z])
    return pt, eta, phi

def get_px_py_pz_v(pt, eta, phi):
    '''Provides px, py, pz, given pt, eta, and phi'''
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    return px, py, pz

def rotate_v(py, pz, angle):
    '''Rotates vector by angle provided'''
    pyy = py * np.cos(angle) - pz * np.sin(angle)
    pzz = pz * np.cos(angle) + py * np.sin(angle)
    return pyy, pzz

# Train data
# Read in hdf5 files
# Note: you will need to repeat this processing for input_filenames train, test and val


for input_filename in ["raw/train.h5", "raw/test.h5", "raw/val.h5"]:
    assert os.path.exists(input_filename)
    assert input_filename.endswith(".h5")
    store = pd.HDFStore(input_filename)
    df = store.select("table")

    n_constits = 60 # use only 60 highest pt jet constituents 
    df_pt_eta_phi = pd.DataFrame()
    
    e_cols = ["E_{}".format(i) for i in range(n_constits)]
    px_cols = ["PX_{}".format(i) for i in range(n_constits)]
    py_cols = ["PY_{}".format(i) for i in range(n_constits)]
    pz_cols = ["PZ_{}".format(i) for i in range(n_constits)]
    jet_e = np.array(df[e_cols].sum(axis=1)).reshape(-1,1)
    jet_px = np.array(df[px_cols].sum(axis=1)).reshape(-1,1)
    jet_py = np.array(df[py_cols].sum(axis=1)).reshape(-1,1)
    jet_pz = np.array(df[pz_cols].sum(axis=1)).reshape(-1,1)
    jet_pt, jet_eta, jet_phi = get_pt_eta_phi_v(jet_px.flatten(), jet_py.flatten(), jet_pz.flatten())
    jet_pt = jet_pt.reshape(-1,1)
    jet_eta = jet_eta.reshape(-1,1)
    jet_phi = jet_phi.reshape(-1,1)
    jet_m = (np.abs(jet_e**2 - jet_px**2 - jet_py**2 -jet_pz**2))**0.5
    jet_nconst = np.array((df[e_cols] != 0).sum(axis=1)).reshape(-1,1)
    for j in range(n_constits):
        i = str(j)
        #print("Processing constituent #"+str(i))
        px = np.array(df["PX_"+i][0:])
        py = np.array(df["PY_"+i][0:])
        pz = np.array(df["PZ_"+i][0:])
        pt,eta,phi = get_pt_eta_phi_v(px,py,pz)
        df_pt_eta_phi_mini = pd.DataFrame(np.stack([pt,eta,phi]).T,columns = ["pt_"+i,"eta_"+i,"phi_"+i])
        df_pt_eta_phi = pd.concat([df_pt_eta_phi,df_pt_eta_phi_mini], axis=1, sort=False)
    df_pt_eta_phi = df_pt_eta_phi.astype('float32')   
    eta_cols = [col for col in df_pt_eta_phi.columns if 'eta' in col]
    phi_cols = [col for col in df_pt_eta_phi.columns if 'phi' in col]
    pt_cols = [col for col in df_pt_eta_phi.columns if 'pt' in col]
    labels = np.expand_dims(df["is_signal_new"].to_numpy(), axis=0)
    labels = np.append(1-labels, labels, 0)
    
    del df

    jet_ptsum = df_pt_eta_phi[pt_cols].sum(axis=1).to_numpy().reshape(-1,1)
    df_pt_eta_phi[pt_cols]= df_pt_eta_phi[pt_cols].div(df_pt_eta_phi[pt_cols].sum(axis=1), axis=0)
    df_pt_eta_phi[eta_cols] = df_pt_eta_phi[eta_cols].subtract(pd.Series(jet_eta.flatten()),axis=0)
    df_pt_eta_phi[phi_cols] = df_pt_eta_phi[phi_cols].subtract(pd.Series(jet_phi.flatten()),axis=0)
    
    aug_data = np.concatenate((jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst), axis = 1)
    
    data = df_pt_eta_phi.to_numpy()
    mask = np.where(df_pt_eta_phi[pt_cols].to_numpy() != 0, 1, 0)

    labels = np.transpose(labels, (1, 0))
    
    mask = np.reshape(mask, (-1, 1, n_constits))
    data = np.reshape(data, (-1, n_constits, 3))
    #gets rid of phi/eta subtraction from zero-vectors
    m=np.reshape(mask, (-1, n_constits, 1))
    data = np.concatenate([m, m, m], axis=2) * data
    
    new_dataset = h5py.File(input_filename.replace("raw", "processed"), 'w')
    new_dataset.create_dataset('particles', data = data)
    new_dataset.create_dataset('masks', data = mask)
    new_dataset.create_dataset('labels', data = labels)
    new_dataset.create_dataset('aug_data', data = aug_data)

    new_dataset.close()