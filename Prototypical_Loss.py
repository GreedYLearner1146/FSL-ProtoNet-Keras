# Compute the prototypical loss.

def prototypical_loss(prototypes, query_embeddings, query_labels, unique_labels):
    dists = tf.norm(tf.expand_dims(query_embeddings, axis=1) - prototypes, axis=-1)
    labels = tf.cast(tf.equal(tf.expand_dims(query_labels, axis=1), unique_labels), tf.float32)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=-dists))
