# Logs

## To do

- [ ] Loading data (4)
  - [X] ~~Generate graph for the training ?~~ (4.1) see `data.random_graph_generator`
  - [ ] ~~Create a On-The-Fly DataLoader~~ (4.2) see `data.loaders`
  - [X] ~~Generate batch of data with positive and negative examples for a round of training.~~ (4.3) see `data.batchs.augment_batchs`
  - [ ] Use a query over a bigger graph ? (4.4)


- [ ] Having a first training to validate result (6)

- [ ] Check the alligment matrix from the original git (9)

## Doing

- [ ] Create an evaluation methodology (11)
  - [X] Reproduce the Enzyme dataset test
  - [ ] Create a test based on the IBM Dataset (most important)
  - [ ] Create a test based on bigger synthetic Graphs
  > I probably won't have enough time to do them all. 

- [ ] Study the loss and how it can be enhanced
  -[ ] Test different version of the loss and prediction. 


- [ ] See how to adapt NM to directed graph 

  - [ ] Study Directed-GNN 
  - [ ] Adapt proof for the order embedding 
  - [ ] Compare resuts over directed graphs given directed query in multiple situation


- [ ] Having a first training to validate result (6)
  - [X] ~~create a test evaluation to draw Prec-Recall Curve~~
  - [ ] Redo exactly what is indicating in the paper to see if the result match. 

## Done

* [X] Generating synthetic graph for the training (4.1) (05/07/2024-...)
  * ~~Creating generators with different models~~ (`random_graph_generator.py`) (05/14)
  * ~~Generating graphs batch throught Data Loaders iterators~~ (05/22)
  * ~~Augmenting Data to add features~~


* [X] Implementing neuromatch with SAGE (1.1) (05/02/2024 - ...)
  * [X] ~~Creating the neural network used in the initial training (`NM.NeuroMatchNetwork`) with SAGE layer.~~ (05/14)
  * [X] ~~Implementing the skip among the GNN convolutional Sage layers.~~

- [X] Validation function (3)
  - [X] ~~Implementing prediction function~~ `NM.(NeuroMatchPred).treshold_predict`
  - [X] ~~Checking what threshold is used in the original git~~ They used a Linear ML classification in place of a treshold to directly adapt the model. see `NM.NeuroMatchPred` class. 
  - [X] ~~Computing accuracy and loss~~ loss : `NM.NeuroMatchPred.hinge_loss(cls)`, `nmPred.predict(self)`

- [X] Implementing the full training process (5)

  - [X] ~~Handling the optimizer and scheduler~~ *The scheduler actually doesn't exist in classic version*
  - [X] ~~Creating the subprocess function~~
  - [X] ~~Dealing with verification~~
  - [X] ~~Dealing with the training loop~~
  - [X] ~~Dealing with the multiprocessing~~


- [X] Implement a validation function (8)

  - [X] ~~Studying the validation function of the original git~~
  - [X] ~~What is test_pts ?~~ Same as any other batchs see `evaluation.generating_evaluation_batchs`. 
  - [X] ~~Recoding interesting part~~
  - [X] ~~Code a final function training_test~~ see `evaluation.training_test`
  - [X] ~~Testing~~ 
  - [X] ~~Implementing in training_loop~~

- [X] Checkpoints and save system (7)
  - [X] ~~Saving the model (on every epoch probably)~~ Save every 5 epoch
  - [X] ~~Loading from a saved model~~ see the Test notebook > Section Precision Recall curves

- [X] Finding solution to the loss spike (10)
  - [X] ~~Adding a scheduler.~~ Scheduler applicated on each worker for each dataloader (of 1000 batchs) alone , see run `Jun15_10-32-...`
      > More stable but other training seems to indicate it would be possible to have a better accuracy. Way better.  
    - [X] ~~Scheduling on every full epoch better than on mini-batch scale.~~ Scheduler applicated uniformely for every worker on each epochs (5000 batchs) see run `Jun17_15-35-...`
      > The loss spike are still present, even if the loss seems to be frozed on a lesser value. The problem is that epoch are too long and the lr would need to be updated sooner. 
    - [X] ~~Scheduling on every batch but uniformely for every workers.~~ see run `Jun17_16-24...` with epochs of 5000 batchs, learning rate `5e-5`, and a minimum lr of `1e-9`. 
      > The accuracy and losses are way better. Whether for the embedding loss or the prediction loss, they still are very 

  - [X] ~~Changes total batch evaluated (MiniBatch numbers modif)~~
    > I've changed epoch interval length and the number of batch for each worker (mini_batch interval) to reduced the overall epoch to 2000 batchs (before that it was 1e6 batchs)
  - [X] ~~Correction of neuromatch criterion false implementation.~~ `Jun17-19-44`
  - [X] ~~Changing the post-mp network to reduce the length of the full NN.~~  see test when deleting last part of post-nn in `Jun18-10-35` to be compared with `Jun17-19-44`
  - [X] Reducing starting lr 
    > When reduced half, the spikes are less more present even if the training is still very noisy. 