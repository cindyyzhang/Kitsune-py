from Kitsune import Kitsune
import numpy as np
import time
import csv

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# print("Unzipping Sample Capture...")
# import zipfile
# with zipfile.ZipFile("mirai.zip","r") as zip_ref:
#     zip_ref.extractall()

from parse_args import * 

args = parse_args()
dataset = args.dataset
desc = args.job_description


# File location
path = f"{dataset}_pcap.pcap" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# Get labels
labels = f"{dataset}_labels.csv"  #the labels for the pcap packet data
with open(labels, 'r') as f:
    reader = csv.reader(f)
    labels_list = list(reader)
labels_list = [int(sublist[-1]) for sublist in labels_list] #flatten list of labels
if dataset != 'mirai':
    if '0' not in labels_list[0]: 
        labels_list.pop(0)

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 10000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 100000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()

normal_rmses = []
normal_indices = []
anomaly_rmses = []
anomaly_indices = []

# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
all_vec = []
while True:
    i+=1
    if i % 1000 == 0:
        print("packet ", i, " rmse ", RMSEs[-1])
    rmse, vec = K.proc_next_packet()
    if vec is not None:
        all_vec.append(vec)
    if rmse == -1:
        break
    if labels_list[i-1] == 0:
        normal_rmses.append(rmse)
        normal_indices.append(i)
    else:
        anomaly_rmses.append(rmse)
        anomaly_indices.append(i)
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

normal_rmses = np.array(normal_rmses)
normal_indices = np.array(normal_indices)
anomaly_rmses = np.array(anomaly_rmses)
anomaly_indices = np.array(anomaly_indices)
all_vec = np.array(all_vec)
np.save(f"results/{dataset}_{desc}_normal_rmses.npy", normal_rmses)
np.save(f"results/{dataset}_{desc}_normal_indices.npy", normal_indices)
np.save(f"results/{dataset}_{desc}_anomaly_rmses.npy", anomaly_rmses)
np.save(f"results/{dataset}_{desc}_anomaly_indices.npy", anomaly_indices)
np.savetxt(f"results/{dataset}_{desc}_all_vec.csv", all_vec)
print('All vectors saved, shape', all_vec.shape)

