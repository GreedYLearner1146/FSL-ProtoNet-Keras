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

# Training loop
for epoch in range(num_epochs):
    loss = train_step(feature_extractor, support_images, support_labels, query_images, query_labels, optimizer)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}")



