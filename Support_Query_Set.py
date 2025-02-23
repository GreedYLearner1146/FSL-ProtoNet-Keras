############## Support and Query (TRAINING CLASS) ##################################

unique_classes = np.unique(y_train_fs)

# Select n_way classes out of the unique classes.
selected_classes = np.random.choice(unique_classes, n_way, replace=False)

support_set = []
query_set = []

for cls in selected_classes:

    # Get all examples for the current class
    cls_indices = np.where(y_train_fs == cls)[0]  # The indices of the image array where the condition is met.
    np.random.shuffle(cls_indices)  # Shuffle the indices.

    # Select support and query examples
    support_indices = cls_indices[:k_shot]  # First n_way as the support.
    query_indices = cls_indices[k_shot:k_shot + query_samples_per_class]  # Remaining n_way*k_shot as the query.

    # Support and query as tuple (support_img, support_label) , (query_img, query_label)
    support_set.append((x_train_fs[support_indices], np.full(k_shot, cls)))              # Assigned same class label to selected support samples.
    query_set.append((x_train_fs[query_indices], np.full(query_samples_per_class, cls))) # Assigned same class label to selected query samples.

# Concatenate support and query sets
support_images = np.concatenate([x for x, _ in support_set]) # Extract support img.
support_labels = np.concatenate([y for _, y in support_set]) # Extract support label.
query_images = np.concatenate([x for x, _ in query_set])     # Extract query img.
query_labels = np.concatenate([y for _, y in query_set])     # Extract query label.


# Writing the above codes as a function:

def create_episode(x_data, y_data, n_way, k_shot, query_samples_per_class):
    """
    Create a few-shot episode.
    Args:
        x_data: Input data (images).
        y_data: Labels.
        n_way: Number of classes in the episode.
        k_shot: Number of support examples per class.
        query_samples_per_class: Number of query examples per class.
    Returns:
        support_set: Support set images and labels.
        query_set: Query set images and labels.
    """
    unique_classes = np.unique(y_data)
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)

    support_set = []
    query_set = []

    for cls in selected_classes:
        
        # Get all examples for the current class
        cls_indices = np.where(y_data == cls)[0]
        np.random.shuffle(cls_indices)

        # Select support and query examples
        support_indices = cls_indices[:k_shot]
        query_indices = cls_indices[k_shot:k_shot + query_samples_per_class]

        support_set.append((x_data[support_indices], np.full(k_shot, cls)))

        query_set.append((x_data[query_indices], np.full(query_samples_per_class, cls)))

    # Concatenate support and query sets

    support_images = np.concatenate([x for x, _ in support_set])
    support_labels = np.concatenate([y for _, y in support_set])
    query_images = np.concatenate([x for x, _ in query_set])
    query_labels = np.concatenate([y for _, y in query_set])

    return (support_images, support_labels), (query_images, query_labels)

########################################## Support and Query (VALIDATION CLASS) ############################################################## 

unique_val_classes = np.unique(y_val_fs)
print(unique_val_classes)

# Select 5 classes out of the unique classes.
selected_val_classes = np.random.choice(unique_val_classes, n_way, replace=False)
print(selected_val_classes)

support_val_set = []
query_val_set = []

for cls in selected_val_classes:
    print(cls)
    # Get all examples for the current class
    cls_indices = np.where(y_val_fs == cls)[0]  # The indices of the image array where the condition is met.
    np.random.shuffle(cls_indices)
    print(cls_indices)

    # Select support and query examples
    support_indices = cls_indices[:k_shot]  # First 5 as the support.
    print(support_indices)
    query_indices = cls_indices[k_shot:k_shot + query_samples_per_class]  # Remaining 15 as the query.
    print(query_indices)

    # Support and query as tuple (support_img, support_label) , (query_img, query_label)
    support_val_set.append((x_val_fs[support_indices], np.full(k_shot, cls)))              #Assigned same class label to selected support samples.
    query_val_set.append((x_val_fs[query_indices], np.full(query_samples_per_class, cls))) # Assigned same class label to selected query samples.

# Concatenate support and query sets
support_val_images = np.concatenate([x for x, _ in support_val_set]) # Extract support img.
support_val_labels = np.concatenate([y for _, y in support_val_set]) # Extract support label.
query_val_images = np.concatenate([x for x, _ in query_val_set])     # Extract query img.
query_val_labels = np.concatenate([y for _, y in query_val_set])     # Extract query label.

############################################################################################################################################################ 

(support_set, support_labels), (query_set, query_labels) = create_episode(x_train_fs, y_train_fs, n_way, k_shot, query_samples_per_class) # training set
(support_val_set, support_val_labels), (query_val_set, query_val_labels) = create_episode(x_val_fs, y_val_fs, n_way, k_shot, query_samples_per_class) # Valid set

print("Support set shape:", support_set.shape)
print("Query set shape:", query_set.shape)

print("Support val set shape:", support_val_set.shape)
print("Query val set shape:", query_val_set.shape)
