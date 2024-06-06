"""training module for the neuromatch GNN
"""

# ---- Import library ----
import torch
import torch.multiprocessing as mp
import torch.nn as nn


import data.random_graph_generator as rgg
from data.loaders import gen_data_loaders
from data.batchs import augment_batch
from models.NM import NeuroMatchNetwork, nm_criterion
# ------      ------

def train(model:NeuroMatchNetwork,generator:rgg.RandomGenerator,in_queue:mp.Queue,out_queue:mp.Queue):
    done = False
    while not done :
        x = in_queue.get()
        if x == "done" :
            done = True
            break

        loaders = gen_data_loaders(generator=generator)
        # --- Training ---
        for pos_batch, neg_batch, _ in zip(*loaders) :
            model.train()
            model.zero_grad()
            pos_target, pos_query, neg_target, neg_query = augment_batch(pos_batch,neg_batch,generator)
            emb_pos_tar = model(pos_target)
            emb_pos_que = model(pos_query)
            emb_neg_tar = model(neg_target)
            emb_neg_que = model(neg_query)

            loss = nm_criterion(emb_pos_tar,emb_pos_que,emb_neg_tar,emb_neg_que)
            loss.backward()
            opt.step()
            scheduler.step()

            with torch.no_grad():
                pred = predict(emb_target,emb_query)
            model.clf_model.zero_grad()
            pred = model.clf_model(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            
            out_queue.put(("step", (loss.item(), acc)))