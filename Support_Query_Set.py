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
