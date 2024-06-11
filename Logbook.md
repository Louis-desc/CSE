# Logs

## To do

- [ ] Loading data (4)
  - [X] Generate graph for the training ? (4.1)
  - [ ] Use a query over a bigger graph ? (4.2)
  - [X] Generate batch of data with positive and negative examples for a round of training. (4.3)


- [ ] Having a first training to validate result (6)

- [ ] Checkpoints and save system (7)
  - [ ] Saving the model (on every epoch probably)
  - [ ] Loading from a saved model 

- [ ] Implement a validation function (8)

- [ ] Check the alligment matrix from the original git (9)

## Doing

- [ ] Implement a validation function (8)


## Done

* [X] Generating synthetic graph for the training (4.1) (05/07/2024-...)
  * ~~Creating generators with different models~~ (`random_graph_generator.py`) (05/14)
  * ~~Generating graphs batch throught Data Loaders iterators~~ (05/22)
  * ~~Augmenting Data to add features~~

- [ ] Loading data (4)

  - [X] Generate graph for the training ? (4.1)
  - [ ] Use a query over a bigger graph ? (4.2)
  - [X] Generate batch of data with positive and negative examples for a round of training. (4.3)
- [X] Implementing the loss functions according to NeuroMatch paper (2) `NM.nm_criterion`

* [X] Implementing neuromatch with SAGE (1.1) (05/02/2024 - ...)
  * [X] ~~Creating the neural network used in the initial training (`NM.NeuroMatchNetwork`) with SAGE layer.~~ (05/14)
  * [X] ~~Implementing the skip among the GNN convolutional Sage layers.~~

- [X] Validation function (3)

  - [X] ~~Implementing prediction function~~ `NM.nm_predict`
  - [X] ~~Checking what threshold is used in the original git~~ They used a Linear ML classification in place of a treshold to directly adapt the model.
  - [X] ~~Computing accuracy and loss~~

- [X] Implementing the full training process (5)

  - [X] ~~Handling the optimizer and scheduler~~ *The scheduler actually doesn't exist in classic version*
  - [X] ~~Creating the subprocess function~~
  - [X] ~~Dealing with verification~~
  - [X] Dealing with the training loop
  - [X] Dealing with the multiprocessing