"""training module for the neuromatch GNN
"""

# ---- Import library ----
import os
import sys
import copy
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import data.random_graph_generator as rgg
from utils.torch_ml import get_device
from data.loaders import gen_data_loaders
from data.batchs import augment_batch
from data.real_graph_generator import RealGenerator
from models.NM import NeuroMatchNetwork, NeuroMatchPred, nm_criterion, faulty_criterion
from evalutation import training_test, generating_evaluation_batchs
# ------      ------

# ---- Constants ----
MODEL_PATH = "ckpt/"
GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 35
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4                         #Original : 1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = True


# ---- Functions ----

def train(model:NeuroMatchNetwork,prediction_model:NeuroMatchPred,generator:rgg.Generator,in_queue:mp.Queue,out_queue:mp.Queue):
    """Algorithm of workers. This functions mostly deals with training the embedding models 
    (see NeuroMatchPred.train(), which is called here, for details on prediction training). 

    Parameters
    ----------
    model : NeuroMatchNetwork
        Instance of a Neuromatch based embedding model. 
    prediction_model : NeuroMatchPred
        Instance of a Neuromatch based prediction model.
    generator : rgg.Generator
        A generator that will be used for creating batches.
    in_queue : mp.Queue
        Multiprocessing in-queue
    out_queue : mp.Queue
        Multiprocessing out-queue

    Raises
    ------
    ChildProcessError
        A wrong order have been given by the parent program of this worker.
    """
    done = False

    lr = LR #Initialization
    filter_fn = filter(lambda p : p.requires_grad, model.parameters())
    opt = optim.Adam(filter_fn, lr, weight_decay=WEIGHT_DECAY)

    while not done :

        loaders = gen_data_loaders(mini_epoch=1000, generator=generator)
        #The loader should behave as if it was an infinite generator. I could probably modify that in the future.
        # --- Training ---
        for pos_batch, neg_batch, _ in zip(*loaders) :
            # -- Waiting for parent process to assign a step --
            x = in_queue.get()
            if x == "done" :
                done = True
                break
            if x[1]=="step" :
                if x[0]!= lr: #Checking if lr was modified
                    lr = x[0]
                    filter_fn = filter(lambda p : p.requires_grad, model.parameters())
                    #We need to redefine the filter every minibatch to redefine the optimizer
                    opt = optim.Adam(filter_fn, lr, weight_decay=WEIGHT_DECAY)
                else:
                    pass #if lr wasn't modified, it doesn't raise exception and continue
            else :
                raise ChildProcessError
            # ---

            model.train()
            model.zero_grad()
            pos_target, pos_query, neg_target, neg_query = augment_batch(pos_batch,neg_batch,generator)
            emb_pos_tar = model(pos_target)
            emb_pos_que = model(pos_query)
            emb_neg_tar = model(neg_target)
            emb_neg_que = model(neg_query)

            if CLASSIC_LOSS :
                loss = nm_criterion(emb_pos_tar,emb_pos_que,emb_neg_tar,emb_neg_que)
            else:
                loss = faulty_criterion(emb_pos_tar,emb_pos_que,emb_neg_tar,emb_neg_que)
            loss.backward()
            opt.step()
            # scheduler.step(loss)  Run Jun15_10_32

            # --- Verification of the model using ML Prediction ---
            emb_target = torch.cat((emb_pos_tar, emb_neg_tar), dim=0)
            emb_query = torch.cat((emb_pos_que, emb_neg_que), dim=0)
            labels = torch.tensor([1]*32 + [0]*32).to(get_device())
            pred, pred_loss = prediction_model.train(emb_target,emb_query,labels)
            pred = pred.argmax(dim=-1)
            pred_acc = torch.mean((pred == labels).type(torch.float))

            out_queue.put(("step", (loss.item(), pred_acc, pred_loss)))
    sys.exit(0)

def train_loop(dataset_str:str ="synthetic"):
    """Training process of the embedding and the prediction algorithm together. 
    This functions mainly handle multiprocessing, scheduler and epochs verification. 
    Two other functions (see `train()` for embeddingand `NeuroMatchPred.train()` for prediction ) 
    handle the loss and mini batchs. 
    
    The single functions parameters change the type of dataset to train on. 
    However, Global variables of this module (train) are to change for some parameters on the training.
    (For example what loss to use, Scheduler, Starting loss value, number of workers, ...)

    Parameters
    ----------
    dataset_str : str, optional
        Name of the TUDataset to train on (e.g. ENZYMES, COX2, ...) or "synthetic" for randomly generated graphs , by default "synthetic"
    """
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH))

    # --- Local Params
    in_queue, out_queue = mp.Queue(), mp.Queue()

    emb_model = NeuroMatchNetwork()
    pred_model = NeuroMatchPred()
    emb_model.share_memory()
    pred_model.model.share_memory()
    # --- Choosing dataset type
    if dataset_str =="synthetic" :
        generator = rgg.random_generator(GRAPH_SIZES)
    else:
        print(f"using TUDataset : {dataset_str}")
        generator = RealGenerator(dataset_str)
    writer = SummaryWriter()

    #evaluation batch
    batchs_list = generating_evaluation_batchs(generator)

    # --- Creating parrallel workers
    workers = []
    for i in range(N_WORKERS):
        worker = mp.Process(target=train, args=(emb_model,pred_model,copy.deepcopy(generator),in_queue,out_queue))
        worker.start()
        print(f"Worker {i} PID : {worker.pid}")
        workers.append(worker)

    # --- initializing learning rate, optimizer and scheduler for embedding (to be passed to worker each time)
    filter_fn = filter(lambda p : p.requires_grad, emb_model.parameters())
    lr = LR
    if SCHEDULER_ON:
        opt = optim.Adam(filter_fn, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt,patience=1500)

    # --- Training by epochs
    batch_n = 0
    for epoch in range(EPOCHS):
        for _ in range(EPOCH_INTERVAL):
            in_queue.put((lr,"step"))

        for _ in range(EPOCH_INTERVAL): #An epoch correspond to 1000 batchs in this disposition
            _, (loss, acc,pred_loss) = out_queue.get()
            print(f"Epoch : {epoch:04d}, Batch : {batch_n:06d}, loss : {loss:.4f}, prediction_acc : {acc:.4f}", end="           \r")
            writer.add_scalar("predict/accuracy",acc,batch_n)
            writer.add_scalar("predict/loss",pred_loss,batch_n)
            writer.add_scalar("embedding/loss",loss,batch_n)
            batch_n += 1
            if SCHEDULER_ON:
                scheduler.step(loss)           # The Scheduler is monitoring embedding loss on every
                lr = scheduler.get_last_lr()[-1]    # updating the learning rate if needed
            writer.add_scalar("learning rate", lr,batch_n)

        print(" "*100 ,end="\r")        # Cleaning the line
        training_test(emb_model,pred_model,batchs_list,writer,epoch)

        #Saving models (only every 5 epochs)
        if (epoch+1)%5 == 0:
            torch.save(emb_model.state_dict(), MODEL_PATH+f"emb_model_{epoch}.pt")
            torch.save(pred_model.model.state_dict(), MODEL_PATH+f"pred_model_linear_{epoch}.pt")
        print("-"*50)

    # --- Ending the training
    for i in range(N_WORKERS):
        in_queue.put("done")
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    train_loop()