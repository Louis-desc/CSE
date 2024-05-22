# Logs

## To do

- [ ] Implementing GNN models (1)
  - [ ] Implementing neuromatch with SAGE Model (1.1)
- [ ] Implementing the loss functions according to NeuroMatch paper (2)
- [ ] Validation function (3)
- [ ] Loading data (4)
  - [ ] Generate graph for the training ? (4.1)
  - [ ] Use a query over a bigger graph ? (4.2)
  - [ ] Generate batch of data with positive and negative examples for a round of training. (4.3)
- [ ] Implementing the full training process (5)

## Doing

* [ ] Implementing neuromatch with SAGE (1.1) (05/02/2024 - ...)
  * ~~Creating the neural network used in the initial training (`NM.NeuroMatchNetwork`) with SAGE layer.~~ (05/14)
  * Implementing the skip among the GNN convolutional Sage layers. 
* [ ] Generating synthetic graph for the training (4.1) (05/07/2024-...)
  * ~~Creating generators with different models (`random_graph_generator.py`).~~ (05/14)
  * Generating graphs batch throught Data Loaders iterators 
  * Augmenting Data to add features 
  

## Done
