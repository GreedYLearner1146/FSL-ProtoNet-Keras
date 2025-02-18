# FSL-ProtoNet-Keras
This is the code repository for Few-Shot Prototypical Network in Keras for CIFAR-FS Dataset.
A complementary medium article for Few-Shot Prototypical Learning in Keras will be available soon.

# Code Instructions

- Run Data_loading.py which load the CIFAR-100 dataset and split them into the meta-training, valid and test dataset to form the CIFAR-FS dataset.
- Run Hyperparameters.py which specify the important hyperparameters for the training, for instance the n_way, k_shot, queries_samples_per_class, and epoch.
- Run Support_Query_Set.py which store the support set and the query set, each comprising of the (img, label) tuple-like configuration.
