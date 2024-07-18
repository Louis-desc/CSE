## Done

* [X] Generating synthetic graph for the training (4.1)

  * ~~Creating generators with different models~~ (`random_graph_generator.py`)
  * ~~Generating graphs batch throught Data Loaders iterators~~
  * ~~Augmenting Data to add features~~
* [X] Implementing neuromatch with SAGE (1.1)

  * [X] ~~Creating the neural network used in the initial training (`NM.NeuroMatchNetwork`) with SAGE layer.~~
  * [X] ~~Implementing the skip among the GNN convolutional Sage layers.~~

- [X] Validation function (3)

  - [X] ~~Implementing prediction function~~ `NM.(NeuroMatchPred).treshold_predict`
  - [X] ~~Checking what threshold is used in the original git~~ They used a Linear ML classification in place of a treshold to directly adapt the model. see `NM.NeuroMatchPred` class.
  - [X] ~~Computing accuracy and loss~~ loss : `NM.NeuroMatchPred.hinge_loss(cls)`, `nmPred.predict(self)`
- [X] Implementing the full training process (5)

  - [X] ~~Handling the optimizer and scheduler~~ *The scheduler actually doesn't exist in original implementation*
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
    >

    - [X] ~~Scheduling on every full epoch better than on mini-batch scale.~~ Scheduler applicated uniformely for every worker on each epochs (5000 batchs) see run `Jun17_15-35-...`
      > The loss spike are still present, even if the loss seems to be frozed on a lesser value. The problem is that epoch are too long and the lr would need to be updated sooner.
      >
    - [X] ~~Scheduling on every batch but uniformely for every workers.~~ see run `Jun17_16-24...` with epochs of 5000 batchs, learning rate `5e-5`, and a minimum lr of `1e-9`.
      > The accuracy and losses are way better. Whether for the embedding loss or the prediction loss, they still are very
      >
  - [X] ~~Changes total batch evaluated (MiniBatch numbers modif)~~

    > I've changed epoch interval length and the number of batch for each worker (mini_batch interval) to reduced the overall epoch to 2000 batchs (before that it was 1e6 batchs)
    >
  - [X] ~~Correction of neuromatch criterion false implementation.~~ `Jun17-19-44`
  - [X] ~~Changing the post-mp network to reduce the length of the full NN.~~  see test when deleting last part of post-nn in `Jun18-10-35` to be compared with `Jun17-19-44`
  - [X] Reducing starting lr

    > When reduced half, the spikes are less more present even if the training is still very noisy.
    >
- [X] Having a first training to validate result (6)

  - [X] ~~create a test evaluation to draw Prec-Recall Curve~~
  - [X] ~~Redo what is indicating in the paper to see if the result match.~~

# Runs Hyper parameters

## Run Jun17-19-44 (neuromatch_scheduler_29_Jun17_19_44)

Neuromatch model with a scheduler 


GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 1000
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-5
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = True

## Run Jun18-11-53 (neuromatch_9_Jun18_11_53) 

Neuromatch Classic model (without scheduler) 


GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 1000
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = False
CLASSIC_LOSS = True
Scheduler : patience = 500

## Run Jun20-19-00 (---)

Basically it is a loss with the faulty loss, it is to be compared with Jun21_14_02

The computed model have been deleted but the progression curves on tensorboard are still in 'savedRun'


GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 1000
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-5
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = False
Scheduler : patience = 500

## Run Jun21_14_02 (faulty_loss_34_Jun21_14_02)


GRAPH_SIZES = np.arange(6, 30)
EPOCHS = 1000
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-5
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = False
Scheduler : patience = 1500, factor = 0.01

## Run Jul08_23_39 (cox2_9_Jul08_23_39)

Trained on COX2 Dataset (using RealGenerator implementation).  **COX2 / CLASSIC LOSS**


GRAPH_SIZES = NOT USED
EPOCHS = 10
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = True
Scheduler : patience = 1500, factor = 0.01

## Run Jul09_10_15 (faulty_cox2_9_Jul09_10_15) 

Trained on COX2 Dataset (using RealGenerator implementation). **COX2 / FAULTY LOSS**


GRAPH_SIZES = NOT USED
EPOCHS = 10
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = True
Scheduler : patience = 1500, factor = 0.01

## Run Jul09_13_25 (enzymes_9_Jul09_13_25) 

GRAPH_SIZES = NOT USED
EPOCHS = 10
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = True
Scheduler : patience = 1500, factor = 0.01

## Run Jul09_14_32 (faulty_enzymes_14_Jul09_14_32)

GRAPH_SIZES = NOT USED
EPOCHS = 15
EPOCH_INTERVAL = 1000           # In number of batchs
N_WORKERS = 8
LR=1e-4
WEIGHT_DECAY = 0.0
SCHEDULER_ON = True
CLASSIC_LOSS = False
Scheduler : patience = 1500, factor = 0.01
