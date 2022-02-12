# Classification of radioactivity signals
Preparation  and labeling of radioactivity signals dataset and training a ML model on the prepared data
This repository is the code of proposed novel strategy to prepare clean labeled neutron/gamma radiations data-sets detected by the EJ276 plastic scintillator. First, a pile-up detection method is implemented and evaluated to clean up the acquired signals. Next, a specific method is proposed to identify and remove the obtained mislabeled samples by Time of Flight (ToF) setup. This specific method depends on the results of ToF and the Tail-to-Total integral ratio (TTTratio) discrimination algorithm. The experiments setup is implemented using californium (252Cf) and cobalt (60Co) as radioactivity sources. Finally, the obtained labeled data is used to train an Artificial Neural Network (ANN). Experimental results show that the proposed model is generic and could be applied to different energy radiations and source types. This model
generalization validates the labels provided by the labeling strategy. The obtained accuracy, False Positive Rate (FPR) and True Positive Rate (TPR) are respectively
equal to 97%, 1.5%, and 92%.

