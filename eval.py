unique_classes = np.unique(y_test_fs)  # For test class in CIFAR-FS.
print(unique_classes)

# Select 5 classes out of the unique test classes.
selected_classes = np.random.choice(unique_classes, n_way, replace=False)
print(selected_classes)

################### Make support and query set test ###########################################################

support_set_test = []
query_set_test = []

for cls in selected_classes:
    print(cls)
    # Get all examples for the current class
    cls_indices = np.where(y_test_fs == cls)[0]  # The indices of the image array where the condition is met.
    np.random.shuffle(cls_indices)
    print(cls_indices)

    # Select support and query examples
    support_indices = cls_indices[:k_shot]  # First 5 as the support.
    print(support_indices)
    query_indices = cls_indices[k_shot:k_shot + query_samples_per_class]  # Remaining 15 as the query.
    print(query_indices)

    # Support and query as tuple (support_img, support_label) , (query_img, query_label)
    support_set_test.append((x_test_fs[support_indices], np.full(k_shot, cls)))              #Assigned same class label to selected support samples.
    query_set_test.append((x_test_fs[query_indices], np.full(query_samples_per_class, cls))) # Assigned same class label to selected query samples.

# Concatenate support and query sets
support_images_test = np.concatenate([x for x, _ in support_set_test]) # Extract support img.
support_labels_test = np.concatenate([y for _, y in support_set_test]) # Extract support label.
query_images_test = np.concatenate([x for x, _ in query_set_test])     # Extract query img.
query_labels_test  = np.concatenate([y for _, y in query_set_test])     # Extract query label.


# Make prediction on test set.

def predict(model, support_images, support_labels, query_images):
    support_embeddings = model(support_images, training=False)
    query_embeddings = model(query_images, training=False)
    prototypes, unique_labels = compute_prototypes(support_embeddings.numpy(), support_labels)
    dists = tf.norm(tf.expand_dims(query_embeddings, axis=1) - prototypes, axis=-1)
    predictions = tf.argmin(dists, axis=1)
    return unique_labels[predictions.numpy()]

test_predictions = predict(feature_extractor, support_images_test, support_labels_test, query_images_test)

# Compute accuracy.

def compute_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

# Compute accuracy on test data. Range between [0,1]. Multiply by 100 to get accuracy percentage value.
test_accuracy = compute_accuracy(test_predictions,query_labels_test)
print("Test Accuracy:", test_accuracy)
