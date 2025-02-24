# For training step. Incorporated model embedding, prototypes computation, and prototypical loss.
# Output is the loss function per epoch.

def train_step(model, support_images, support_labels, query_images, query_labels, optimizer):
    with tf.GradientTape() as tape:
        support_embeddings = model(support_images, training=True)
        query_embeddings = model(query_images, training=True)
        prototypes, unique_labels = compute_prototypes(support_embeddings.numpy(), support_labels)
        loss = prototypical_loss(prototypes, query_embeddings, query_labels, unique_labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

feature_extractor = model  # Whatever feature extractor utilized. Please use only one!
optimizer = keras.optimizers.Adam() # Optimizer

loss = train_step(feature_extractor, support_images, support_labels, query_images, query_labels, optimizer)

def predict(model, support_images, support_labels, query_images):
    support_embeddings = model(support_images, training=False)
    query_embeddings = model(query_images, training=False)
    prototypes, unique_labels = compute_prototypes(support_embeddings.numpy(), support_labels)
    dists = tf.norm(tf.expand_dims(query_embeddings, axis=1) - prototypes, axis=-1)
    predictions = tf.argmin(dists, axis=1)
    return unique_labels[predictions.numpy()]

def compute_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

# Training loop
for epoch in range(num_epochs):
    loss = train_step(feature_extractor, support_images, support_labels, query_images, query_labels, optimizer)
    val_predictions = predict(feature_extractor, support_val_set, support_val_labels, query_val_set) # Validation set
    val_accuracy = compute_accuracy(val_predictions, query_val_labels) # Validation accuracy
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}, Validation Accuracy: {val_accuracy}")



