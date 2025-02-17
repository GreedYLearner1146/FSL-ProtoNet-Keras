def compute_prototypes(support_set, support_labels):
    unique_labels = np.unique(support_labels)
    prototypes = []
    for label in unique_labels:
        class_embeddings = support_set[support_labels == label]
        prototype = np.mean(class_embeddings, axis=0)  # Mean of the class embeddings.
        prototypes.append(prototype)
    return np.array(prototypes), np.array(unique_labels)
