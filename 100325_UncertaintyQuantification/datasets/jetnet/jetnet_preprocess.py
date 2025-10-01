import h5py
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    files = ['g.hdf5', 'q.hdf5', 't.hdf5',  'w.hdf5',  'z.hdf5']
    files = ["raw/" + f for f in files]
    nfiles = len(files)

    for ii, f in enumerate(files):
        F = h5py.File(f)
        
        this_particles = F['particle_features'].__array__()
        this_jets = F['jet_features'].__array__()
        ndata = this_particles.shape[0]
        this_labels = np.zeros((ndata, nfiles))
        this_labels[:, ii] += 1

        if ii == 0:
            particles = this_particles.copy()
            jets = this_jets.copy()
            labels = this_labels.copy()
        else:
            particles = np.append(particles, this_particles, 0)
            jets = np.append(jets, this_jets, 0)
            labels = np.append(labels, this_labels, 0)
            
    del this_particles, this_jets, this_labels

    ndata = particles.shape[0]

    print("Particles: ", particles.shape)
    print("Jets: ", jets.shape)
    print("Labels: ", labels.shape)
    
    # particles have shape (ndata, 30, 4): (eta, phi, pt, mask)
    # jets have shape (ndata, 4): (pt, eta, mass, #particles)

    particles = np.where(particles[:, :, [3]] == 0, 0, particles)

    particles, masks = particles[:,:,[2,0,1]], particles[:,:,[3]].reshape(ndata, 1, -1)
    
    print("Particles: ", particles.shape) # Now should be (ndata, 30, 3)
    print("Masks: ", masks.shape) # Now should be (ndata, 1, 30)

    particles[:, :, 0] = particles[:, :, 0] * jets[:, 0:1]
    jet_ptsum = particles[:, :, 0].sum(-1).reshape(-1,1)
    particles[:, :, 0] = particles[:, :, 0] / jet_ptsum

    # Now make augmented data to have the quantities: jet_e, jet_m, jet_pt, jet_eta, jet_phi, jet_ptsum, jet_nconst

    jet_pt = jets[:, 0].reshape(-1,1)
    jet_m = jets[:, 2].reshape(-1,1)
    jet_eta = jets[:, 1].reshape(-1,1)
    jet_nconst = jets[:, 3].reshape(-1,1)

    jet_mt = np.sqrt( jet_pt ** 2 + jet_m **2 )
    jet_e = jet_mt * np.sinh(jet_eta)

    aug  = np.concatenate((jet_e, jet_m, jet_pt, jet_eta, np.zeros_like(jet_eta), jet_ptsum, jet_nconst), 1)

    print("Particles: ", particles.shape) # Now should be (ndata, 30, 3)
    print("Masks: ", masks.shape) # Now should be (ndata, 1, 30)
    print("Aug: ", aug.shape) # Now should be (ndata, 7)
    print("Labels: ", labels.shape) # Now should be (ndata, 7)

    all_indices = list(range(ndata))
    train_idx, test_idx = train_test_split(all_indices, test_size = 0.2)
    train_idx, val_idx   = train_test_split(train_idx, test_size = 0.25)

    # Saving train data

    train_data, train_mask, train_labels, train_aug = particles[train_idx, :, :], masks[train_idx, :, :], labels[train_idx, :], aug[train_idx, :]
    val_data, val_mask, val_labels, val_aug = particles[val_idx, :, :], masks[val_idx, :, :], labels[val_idx, :], aug[val_idx, :]
    test_data, test_mask, test_labels, test_aug = particles[test_idx, :, :], masks[test_idx, :, :], labels[test_idx, :], aug[test_idx, :]

    train = h5py.File('processed/train.h5', 'w')
    train.create_dataset('particles', data = train_data)
    train.create_dataset('masks', data = train_mask)
    train.create_dataset('labels', data = train_labels)
    train.create_dataset('aug_data', data = train_aug)

    train.close()

    test = h5py.File('processed/test.h5', 'w')
    test.create_dataset('particles', data = test_data)
    test.create_dataset('masks', data = test_mask)
    test.create_dataset('labels', data = test_labels)
    test.create_dataset('aug_data', data = test_aug)

    test.close()

    
    val = h5py.File('processed/val.h5', 'w')
    val.create_dataset('particles', data = val_data)
    val.create_dataset('masks', data = val_mask)
    val.create_dataset('labels', data = val_labels)
    val.create_dataset('aug_data', data = val_aug)

    val.close()

    
    
    
    

    
            