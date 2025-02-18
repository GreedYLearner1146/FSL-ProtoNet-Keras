# FSL-ProtoNet-Keras
This is the code repository for Few-Shot Prototypical Network [1] in Keras for CIFAR-FS Dataset.
A complementary medium article for Few-Shot Prototypical Learning in Keras will be available soon.

The diagram that illustrates the overall working of the few-shot prototypical network, and is extracted from the excellent review paper by Parnami and Lee [2]. 

![image](https://github.com/user-attachments/assets/857c8396-ce62-42d4-a716-187d22427d27)


# Code Instructions

- Run Data_loading.py which load the CIFAR-100 dataset and split them into the meta-training, valid and test dataset to form the CIFAR-FS dataset.
- Run Hyperparameters.py which specify the important hyperparameters for the training, for instance the n_way, k_shot, queries_samples_per_class, and epoch.
- Run Support_Query_Set.py which store the support set and the query set, each comprising of the (img, label) tuple-like configuration.
- Run Compute_prototypes.py which compute the prototypes for each selected class to be used for computing the euclidean distance with the query class embedding.
- Run Prototypical_Loss.py which computes the prototypical loss.
- Run Conv_Embedding.py that contained the code for the Conv4 and ResNet12 feature extractor which are best suited for the CIFAR-FS dataset. Choose one feature extractor for the subsequent training.
- Run training.py to begin the training process.
- Finally, run eval.py to evaluate the trained model on the meta-test dataset.

# Relevant Papers

[1] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for few-shot learning,” Advances in neural information processing systems, vol. 30, 2017.
[2] A. Parnami and M. Lee, “Learning from few examples: A summary of approaches to few-shot learning,” arXiv preprint arXiv:2203.04291, 2022. 
