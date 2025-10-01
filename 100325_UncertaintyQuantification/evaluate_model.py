import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torchinfo import summary
import torch.nn as nn
import argparse, os, json, sys
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
from EvalTools import *
from PFINDataset import PFINDataset, JetClassData
from UQPFIN import UQPFIN as Model
import argparse
import gc
# %env HDF5_USE_FILE_LOCKING=FALSE
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, action="store", dest="outdir", default="results/", help="Output directory for evaluation results" )
    parser.add_argument("--data", type=str, action="store", dest="data_type", default="topdata", choices={"topdata", "jetnet", "JNqgmerged", "jetclass"}, help="Dataset to evaluate on" )
    parser.add_argument("--make-file", action="store_true", dest="make_file", default=False, help="Set this flag when recording evaluation results")
    parser.add_argument("--data-loc", type=str, action="store", dest="data_loc", default="./datasets/", help="Directory for data" )
    parser.add_argument("--modeldir", type=str, action="store", dest="modeldir", default="./trained_models/", help="Directory for trained model parameters" )
    parser.add_argument("--modeldictdir", type=str, action="store", dest="modeldictdir", default="./trained_model_dicts/", help="Directory for trained model metadata" )
    parser.add_argument("--tag", type=str, action="store", dest="tag", default="", help="Optional tag to only store results of certain models with tag in the name" )
    parser.add_argument("--type", type=str, action="store", dest="model_type", default="edl", choices={"edl", "ensemble", "dropout"}, help="Type of model to evaluate" )
    parser.add_argument("--batch-mode", action="store_true", dest="batchmode", default=False, help="Set this flag when running in batch mode to suppress tqdm progress bars")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    makeFile = args.make_file

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    saved_model_loc = args.modeldir
    saved_model_dict_loc = args.modeldictdir
    model_version = "_best"

    dataset = args.data_type
    if args.model_type == "dropout":
        all_models = [f for f in os.listdir(saved_model_loc) if model_version in f and dataset in f and "alldicts" not in f and "_softmax" in f and "_dropout" in f and args.tag in f]
        n = 1
    elif args.model_type == "ensemble":
        all_models = [f for f in os.listdir(saved_model_loc) if model_version in f and "_trial" not in f and dataset in f and "_softmax" in f and "_dropout" not in f and args.tag in f] 
        n = 2
    elif args.model_type == "edl":
        all_models = [f for f in os.listdir(saved_model_loc) if model_version in f and "_trial" not in f and dataset in f and 'softmax' not in f and "20M" not in f and args.tag in f]
        n = 1
        
    all_models = sorted(all_models)
    
    tags = []
    for modelname in all_models:
        tag = modelname.strip().split('_')[-n]
        if tag not in tags:
            tags.append(tag)
    if not len(all_models):
        assert False, "No models found"
    print("Type: ", model_version[1:])
    print("Models:")
    print("\n".join(all_models))
    if args.model_type != "edl":
        print("Tags:")
        print("\n".join(tags))
    
    question = "Would you like to proceed with the above models for evaluation using the \033[31m{}\033[0m model on the \033[31m{}\033[0m dataset?".format(args.model_type, dataset)
    answer = query_yes_no(question)
    assert answer, "Stopping evaluation"
    
    if dataset != 'jetclass':
        test_path = os.path.join(args.data_loc, dataset, "processed", "test.h5")
        #Loading testing dataset
        test_set = PFINDataset(test_path)
        if args.model_type == "edl":
            testloader = DataLoader(test_set, shuffle=False, batch_size=512, num_workers=1, pin_memory=True, persistent_workers=True)
    else:
        data_path = glob.glob(os.path.join(args.data_loc, "jetclass", "processed", "test_*.h5"))
        test_set = JetClassData(batch_size = 512)
        test_set.set_file_names(file_names = data_path)
    
    if args.model_type == "edl":
        tags = all_models
        
    for tag in sorted(tags):
        model_results = {}
        
        if args.model_type == "edl" and dataset == "jetclass":
            testloader = test_set.generate_data()
        #Creating Evaluator and recording
        if args.model_type == "dropout":
            this_file = [os.path.join(saved_model_loc, f) for f in all_models if tag in f][0]
            evaluator = MCDOEvaluator(this_file, data_type = dataset)
            labels, preds, maxprobs, probs, sums, oods, uncs, aug = evaluator.evaluate(test_set = test_set, aug = True)
        elif args.model_type == "ensemble":
            this_files = [os.path.join(saved_model_loc, f) for f in all_models if tag in f]
            evaluator = EnsembleEvaluator(this_files, data_type = dataset)
            labels, preds, maxprobs, probs, sums, oods, uncs, aug = evaluator.evaluate(test_set = test_set, aug = True)
        elif args.model_type == "edl":
            this_file = os.path.join(saved_model_loc, tag)
            evaluator = ModelEvaluator(this_file)
            labels, preds, maxprobs, probs, sums, oods, uncs, aug, latents = evaluator.evaluate(data_loader = testloader, latent=True, aug=True)
            nparams = sum(p.numel() for p in evaluator.model.parameters())
            
        acc = accuracy_score(labels[~oods], preds[~oods])*100
        if dataset == "topdata":
            probs2=probs
        else:
            skiplabels = np.unique(labels[oods])
            probs2=np.delete(probs, skiplabels, 1)
            
        if probs2.shape[1] == 2:
            probs2 = probs2[:, 1]
            
        auc = roc_auc_score(labels[~oods], probs2[~oods], multi_class='ovo')*100
        
        #Printing accuracy and AUC and storing into dictionary
        if args.model_type == "edl":
            print("{} \t\t Params: {}\t Accuracy: {:.2f}% \t AUC: {:.2f}%".format(evaluator.label, nparams, acc, auc))
            model_results[evaluator.label] = {'labels' : labels, 
                                              'preds': preds, 
                                              'maxprobs': maxprobs,
                                              'sums':sums, 
                                              'oods':oods,
                                              'uncs': uncs,
                                              'probs': probs,
                                              'latents': latents,
                                              'aug': aug}
        else:
            print("{}\t\t Accuracy: {:.2f}% \t AUC: {:.2f}%".format(tag, acc, auc))
            model_results[tag] = {'labels' : labels, 
                                              'preds': preds, 
                                              'maxprobs': maxprobs,
                                              'sums':sums, 
                                              'oods':oods,
                                              'uncs': uncs,
                                              'probs': probs,
                                              'aug': aug}
        #Saving results if makeFile is True   
        if makeFile:
            if args.model_type == "dropout":
                filename = os.path.join(args.outdir, "RESULTS_UQPFIN_MCDO_" + dataset + "_{}.h5".format(tag))
                k = tag
            elif args.model_type == "ensemble":
                filename = os.path.join(args.outdir, "RESULTS_UQPFIN_Ensemble_" + dataset + "_{}.h5".format(tag))
                k = tag
            elif args.model_type == "edl":
                filename = os.path.join(args.outdir, "RESULTS_{}.h5".format(tag))
                k = evaluator.label
            
            f = h5py.File(filename, "w")
            for key in model_results[k].keys():
                f.create_dataset(key, data = model_results[k][key])
            f.close()
            
            print("Results saved to {}".format(filename))
        del evaluator, model_results, labels, preds, maxprobs, probs, sums, oods, uncs, aug
        torch.cuda.empty_cache()
        gc.collect()