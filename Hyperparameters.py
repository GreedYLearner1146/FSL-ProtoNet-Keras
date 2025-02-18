# Specify the number of ways (class), and the number of support and query shot.
# For reference, the query sample per episodic training is n*way*k_shot

############## For 1-shot learning ####################
n_way = 5   # Number of class revealed per episode
k_shot = 1  # Number of support shot.
query_samples_per_class = 5 # Number of query samples per class.

################ For 5-shot learning #####################

n_way = 5   # Number of class revealed per episode
k_shot = 5  # Number of support shot.
query_samples_per_class = 25 # Number of query samples per class.

num_epochs = 100 # Number of Epoch.
