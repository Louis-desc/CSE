"""training module for the neuromatch GNN
"""

# ---- Import library ----
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


import data.random_graph_generator as rgg
from utils.torch_ml import get_device
from data.loaders import gen_data_loaders
from data.batchs import augment_batch
from models.NM import NeuroMatchNetwork, NeuroMatchPred, nm_criterion
# ------      ------

# ---- Constants ----
MODEL_PATH = "ckpt/model.pt"
GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 1000
EPOCH_INTERVAL = 1000
N_WORKERS = 6


# ---- Functions ----

def train(model:NeuroMatchNetwork,prediction_model:NeuroMatchPred,generator:rgg.RandomGenerator,in_queue:mp.Queue,out_queue:mp.Queue):
    """To be Modified"""
    done = False
    lr=1e-4
    weight_decay = 0.0

    filter_fn = filter(lambda p : p.requires_grad, model.parameters())
    opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

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

            # --- Verification of the model using Machine Learning Prediction ---
            emb_target = torch.cat((emb_pos_tar, emb_neg_tar), dim=0)
            emb_query = torch.cat((emb_pos_que, emb_neg_que), dim=0)
            labels = torch.tensor([1]*32 + [0]*32).to(get_device())
            pred, pred_loss = prediction_model.train(emb_target,emb_query,labels)
            pred = pred.argmax(dim=-1)
            pred_acc = torch.mean((pred == labels).type(torch.float))

            out_queue.put(("step", (loss.item(), pred_acc, pred_loss)))

def train_loop():
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    # --- Local Params
    in_queue, out_queue = mp.Queue(), mp.Queue()

    emb_model = NeuroMatchNetwork()
    pred_model = NeuroMatchPred()
    emb_model.share_memory()
    pred_model.model.share_memory()
    generator = rgg.random_generator(GRAPH_SIZES)

    writer = SummaryWriter()

    # --- Creating parrallel workers
    workers = []
    for i in range(N_WORKERS):
        worker = mp.Process(target=train, args=(emb_model,pred_model,generator,in_queue,out_queue))
        worker.start()
        print(f"Worker {i} PID : {worker.pid}")
        workers.append(worker)

    # --- Training by epochs
    batch_n = 0
    for epoch in range(EPOCHS):
        for _ in range(EPOCH_INTERVAL):
            in_queue.put("step")

        for _ in range(EPOCH_INTERVAL):
            _, (loss, acc,pred_loss) = out_queue.get()
            print(f"Epoch : {epoch:04d}, Batch : {batch_n:06d}, loss : {loss:.4f}, prediction_acc : {acc:.4f}", end="           \r")
            writer.add_scalar("predict/accuracy",acc,batch_n)
            writer.add_scalar("predict/loss",pred_loss,batch_n)
            writer.add_scalar("embedding/loss",loss,batch_n)
            batch_n += 1

    # --- Ending the training
    for i in range(N_WORKERS):
        in_queue.put("done")
    for worker in workers:
        worker.join()
