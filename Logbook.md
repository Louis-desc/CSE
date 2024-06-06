# Logs

## To do

- [ ] Implementing GNN models (1)
  - [X] Implementing neuromatch with SAGE Model (1.1)
    - Implementing Convolutional layer
- [ ] Validation function (3)
- [ ] Loading data (4)
  - [X] Generate graph for the training ? (4.1)
  - [ ] Use a query over a bigger graph ? (4.2)
  - [X] Generate batch of data with positive and negative examples for a round of training. (4.3)
- [ ] Implementing the full training process (5)
- [ ] Having a first training to validate result (6)

## Doing

* [ ] Implementing neuromatch with SAGE (1.1) (05/02/2024 - ...)
  * ~~Creating the neural network used in the initial training (`NM.NeuroMatchNetwork`) with SAGE layer.~~ (05/14)
  * Implementing the skip among the GNN convolutional Sage layers. 


- [ ] Implementing the full training process (5)
  - [ ] Handling the optimizer and scheduler 
  - [X] Creating the subprocess function
  - [ ] Generating the multiprocessing

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