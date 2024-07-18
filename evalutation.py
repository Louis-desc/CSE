"""Evaluating the result of a given NeuroMatch Model. Typically during the training. 
"""

# --- Import ---
from typing import List,Tuple
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, mean_squared_error, precision_recall_curve
from torch_geometric.datasets import TUDataset
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

from data.batchs import sample_subgraph
from data.batchs import augment_batch
from data.batchs import gen_anchor_feature
from data.loaders import gen_data_loaders
from data.random_graph_generator import random_generator,RandomGenerator
from data.dataset import SyntheticDataset
from utils.torch_ml import get_device,to_pyg_data
from models.NM import NeuroMatchPred

# --- Constants ---

GRAPH_SIZES = np.arange(6, 30)
EVAL_SIZE = 64

# --- Functions ---

def training_test(emb_model:nn.Module, pred_model:NeuroMatchPred, batchs_list:list, 
                  writer:SummaryWriter, epoch:int)-> Tuple[torch.Tensor,torch.Tensor]:
    """Training validation routine for each epoch"""
    emb_model.eval()
    pred_model.model.eval()

    print(f"Training Evaluation : Epoch {epoch}")
    pred, labels, e_score = evaluation_predict(emb_model,pred_model,batchs_list)


    # Embedding metrics
    max_loss = torch.max(e_score)
    mean_loss = torch.mean(e_score)
    min_loss = torch.min(e_score)                   #Not sure it's useful bc it should always be 0
    med_loss = torch.median(e_score)
    std_loss = torch.std(e_score)
    mse_loss = mean_squared_error(e_score[labels],torch.zeros_like(e_score[labels]))

    writer.add_scalar("eval/embedding/Max", max_loss, epoch)
    writer.add_scalar("eval/embedding/Min", min_loss, epoch)
    writer.add_scalar("eval/embedding/Mean", mean_loss, epoch)
    writer.add_scalar("eval/embedding/Median", med_loss, epoch)
    writer.add_scalar("eval/embedding/StandardDev", std_loss, epoch)
    writer.add_scalar("eval/embedding/MSE", mse_loss, epoch)

    print(f"Embedding Metrics  : Max : {max_loss}, Mean : {mean_loss}, Standard Deviation : {std_loss}, MSE : {mse_loss}")

    # Prediction metrics
    acc = torch.mean((pred == labels).type(torch.float)).item()
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))                              #Precision over the full batch
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))                                                       #Recall over the full batch

    auroc = roc_auc_score(labels, pred)                                 #sklearn.metrics.roc_auc_score
    avg_prec = average_precision_score(labels, pred)                    #sklearn.metrics.average_precision_score
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()             #sklearn.metrics.confusion_matrix TrueNeg, FalsePos, FalseNeg, TruePos

    writer.add_scalar("eval/Accuracy", acc, epoch)
    writer.add_scalar("eval/Precision", prec, epoch)
    writer.add_scalar("eval/Recall", recall, epoch)
    writer.add_scalar("eval/AUROC", auroc, epoch)
    writer.add_scalar("eval/AvgPrec", avg_prec, epoch)
    writer.add_scalar("eval/TP", tp, epoch)
    writer.add_scalar("eval/TN", tn, epoch)
    writer.add_scalar("eval/FP", fp, epoch)
    writer.add_scalar("eval/FN", fn, epoch)

    print(f"Prediction Metrics : Accuracy : {acc}, Precision : {prec}, Recall : {recall}")
    return mean_loss, acc


def generating_evaluation_batchs(generator=None) ->List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]:
    """Generates a batch list for evaluation purposes.

    Parameters
    ----------
    generator : data.random_graph_generator.Generator, optional
        generator to create the batch, by default random_generator([6,30])

    Returns
    -------
    List[Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]]
        return a list of (pos_target,pos_query,neg_target,neg_query) for each graph. 
        It means len(list) == len(loader) == len(batch).
    """
    batchs_list = []

    if generator is None: #Could do better here
        generator =random_generator(GRAPH_SIZES)

    loaders = gen_data_loaders(mini_epoch=EVAL_SIZE ,generator=generator)

    for pos_batch, neg_batch, _ in zip(*loaders) :
        pos_target, pos_query, neg_target, neg_query = augment_batch(pos_batch,neg_batch,generator)
        pos_target = pos_target.to("cpu")
        pos_query = pos_query.to("cpu")
        neg_target = neg_target.to("cpu")
        neg_query = neg_query.to("cpu")
        batchs_list.append((pos_target,pos_query,neg_target,neg_query))
        log_str = f"Generating testing batch : {len(batchs_list):2d} / {len(loaders[0]):2d}"
        print(log_str, end="      \r")
    print("") #Newline
    return batchs_list


def evaluation_predict(emb_model:nn.Module, pred_model:NeuroMatchPred, batchs_list:list)->Tuple[torch.BoolTensor,torch.BoolTensor,torch.FloatTensor]:
    """Return the list of predictions, labels and the hinge loss for every graphs in the batchs_list 
    using the embedding model and the prediction model

    Parameters
    ----------
    emb_model : nn.Module
        Order Embedding model
    pred_model : NeuroMatchPred
        Prediction model
    batchs_list : list
        the batchs list to predict from. Must be of shape form List[(pos_target,pos_query,neg_target,neg_query)]

    Returns
    -------
    Union[torch.BoolTensor,torch.BoolTensor,torch.FloatTensor]
        returns in this order : 
        1. prediction boolean tensor 
        2. ground truth prediction labels 
        3. hinge loss score (e_score)
    """
    all_e_scores, all_preds, all_labels = [], [], []
    count_loop = 1
    for pos_target,pos_query,neg_target,neg_query in batchs_list:
        print(f"Eval batch : {count_loop} / {len(batchs_list)}", end="        \r")
        labels = torch.tensor([True]*len(pos_target) +
            [False]*len(neg_target)).to(get_device())
        with torch.no_grad() :
            #Embeddings
            emb_pos_tar = emb_model(pos_target)
            emb_pos_que = emb_model(pos_query)
            emb_neg_tar = emb_model(neg_target)
            emb_neg_que = emb_model(neg_query)
            emb_target = torch.cat((emb_pos_tar, emb_neg_tar), dim=0)
            emb_query = torch.cat((emb_pos_que, emb_neg_que), dim=0)

            e_score = NeuroMatchPred.hinge_loss(emb_target,emb_query)
            pred = pred_model.predict(emb_target,emb_query)
            # e_score *= -1         In the original git they do that for an unknown reason

        #Dealing with cache
        count_loop += 1
        all_e_scores.append(e_score.view(e_score.shape[0]))
        all_preds.append(pred)
        all_labels.append(labels)

    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    e_score = torch.cat(all_e_scores, dim=-1)

    pred = pred.detach().cpu()
    labels = labels.detach().cpu()
    e_score = e_score.detach().cpu()

    print(" "*100, end=" \r")

    return pred, labels, e_score


def nm_pr_curve(emb_model:nn.Module, pred_model:NeuroMatchPred, batchs_list:list):
    """Draws the Precision Recall curves for an embedding and a precision model.

    Parameters
    ----------
    emb_model : nn.Module
        Embedding model to evaluate
    pred_model : NeuroMatchPred
        Prediction model to evaluate
    batchs_list : list
        test batch
    """

    emb_model.eval()
    pred_model.model.eval()
    pred, labels, e_score = evaluation_predict(emb_model,pred_model,batchs_list)

    acc = torch.mean((pred == labels).type(torch.float)).item()
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))                              #Precision over the full batch
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))                                                       #Recall over the full batch

    print(f"Prediction Metrics : Accuracy : {acc}, Precision : {prec}, Recall : {recall}")
    precs, recalls, _ = precision_recall_curve(labels, e_score*(-1))       #sklearn.metrics.precision_recall_curve
    plt.plot(recalls, precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("plots/precision-recall-curve.png")
    print("Saved PR curve plot in plots/precision-recall-curve.png")


def verif_test(emb_model:nn.Module, pred_model:NeuroMatchPred, dataset:str="enzyme")-> Tuple[float,float,float]:
    """Test for the Enzyme dataset

    Parameters
    ----------
    emb_model : nn.Module
        Model for embedding graphs
    pred_model : NeuroMatchPred
        Model for predicting over the embedding
    dataset : str
        Can be one of "enzyme" or "synthetic_large", "synthetic_little" dataset. 
        Basically change the type of data. 

    Returns
    -------
    Tuple[float,float,float]
        return scores for the three different subtest. Each score is between 0 and 1, 0 being worst result to 1 perfect result. 
        In this order : 
        - Postive query (sampled from the graph) 
        - Negative Random query (random graph query sample)
    """
    # --- Initialization ---
    # ds = TUDataset(root="../Testds/ENZYMES", name="ENZYMES")
    # generator =random_generator(range(6,126))
    ds, generator = dataset_choice(dataset)
    target, query = [], []
    tries = 10

    for graph in ds:
        if graph.num_nodes < 6 or not nx.is_connected(pyg_utils.to_networkx(graph,to_undirected=True)) :
            # Not dealing with too little graph and unconnected.
            continue
        g_tar, g_quer = sample_subgraph(graph,False)
        target.append(g_tar)
        query.append(g_quer)

    n = len(target)
    # --- Test prediction for positive ---
    pos_query = pyg_data.Batch.from_data_list(query)
    pos_target = pyg_data.Batch.from_data_list(target)
    emb_target = emb_model(pos_target)
    emb_query = emb_model(pos_query)
    sum_pos = sum(pred_model.predict(emb_target,emb_query))
    pos_score = sum_pos/n

    # --- Test differences for randomly given graph ---
    sum_neg_random = 0
    for _ in range(tries) :
        random_query = []
        for k in range(n) :
            g = to_pyg_data(generator.generate(query[k].num_nodes))
            g.x = gen_anchor_feature(g,random.randint(0,g.num_nodes))
            random_query.append(g)

        neg_query = pyg_data.Batch.from_data_list(random_query)
        emb_neg_query = emb_model(neg_query)
        sum_neg_random += sum(~pred_model.predict(emb_target,emb_neg_query))

    score_neg_random = sum_neg_random / (n*tries)

    # --- Test differences when shuffling querries ---
    sum_neg_shuffled = 0
    for _ in range(tries):
        shuffled_emb = emb_query[torch.randperm(emb_query.size()[0])]
        sum_neg_shuffled += sum(~pred_model.predict(emb_target,shuffled_emb))
    score_neg_shuffled = sum_neg_shuffled / (n*tries)

    return pos_score, score_neg_random, score_neg_shuffled


def dataset_choice(mode:str) ->Tuple[pyg_data.Dataset,RandomGenerator]:
    """_summary_

    Parameters
    ----------
    mode : str
        The dataset that is to charge. The possible modes are : 
            - "enzyme"          : ENZYMES Dataset
            - "cox2"            : COX2 Dataset
            - "synthetic_large" : Synthetics graphs of 20 to 90 nodes
            - "synthetic_little": Synthetics graphs of 6 to 30 nodes

    Returns
    -------
    Tuple[pyg_data.Dataset,RandomGenerator]
        Returns a tuple of the requested dataset and a generator that generate *Negative examples*

    Raises
    ------
    NotImplementedError
        In cases the requested dataset is not part of the choices
    """
    if mode == "enzyme" :
        ds = TUDataset(root="../Testds/ENZYMES", name="ENZYMES")
        generator =random_generator(range(6,126)) # Min and Max of Enzyme Dataset
        return ds, generator
    if mode == "synthetic_large" :
        ds = SyntheticDataset(root="../Testds/SYNTHETIC")
        generator = random_generator(ds.graph_size)
        return ds, generator
    if mode == "synthetic_little" :
        ds = SyntheticDataset(root="../Testds/SYNTHETIC_LITTLE",graph_sizes=np.arange(6,30))
        generator = random_generator(ds.graph_size)
        return ds, generator
    if mode == "cox2" : 
        ds = TUDataset(root="../Testds/COX2", name="COX2")
        generator =random_generator(range(32,56)) #Min and max of the Cox2 Dataset
        return ds, generator
    raise NotImplementedError