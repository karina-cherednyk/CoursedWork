import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(tf.cast(real, tf.float32), tf.cast(tf.argmax(pred, axis=-1), tf.float32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def counter(tokens):
    counter1 = {}
    for token in tokens:
        if token == 0:
            continue
        if token in counter1:
            counter1[token] += 1
        else:
            counter1[token] = 1
    return counter1

def bleu_score1(real, pred):
    counter1 = counter(real)
    counter2 = counter(pred)
    res = 0
    for token in counter2:
        if token in counter1:
            res += min(counter1[token], counter2[token])

    return res / len(pred)

def bleu_score2(real, pred):
    real, pred = tf.boolean_mask(real, tf.cast(real, tf.bool)), tf.boolean_mask(pred, tf.cast(pred, tf.bool))
    real_k, _, real_c = tf.unique_with_counts(real)
    pred_k, _, pred_c = tf.unique_with_counts(pred)
    common_elems = tf.where(tf.equal(tf.expand_dims(real_k,1), tf.expand_dims(pred_k,0) ))
    common_real_idx = common_elems[:,0]
    common_pred_idx = common_elems[:,1]
    scores = tf.math.minimum( tf.gather(real_c, common_real_idx), tf.gather(pred_c, common_pred_idx) )
    return tf.cast( tf.reduce_sum(scores) / tf.size(pred), tf.float32) 


def bleu_score(real, pred):
    real = tf.cast(real, tf.float32)
    pred = tf.cast(tf.math.argmax(pred, -1), tf.float32)
    z = tf.stack([real, pred], 1)
    scores = tf.map_fn(lambda x: bleu_score2(x[0], x[1]), z)
    return tf.reduce_sum(scores) / tf.cast(tf.size(scores), tf.float32)
