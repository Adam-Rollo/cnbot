import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class Encoder:
    def __init__(self, embedding, hidden_size, num_layers = 1):
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = rnn.GRUCell(self.hidden_size)
        
    def __call__(self, inputs, seq_length, state=None):
        out = tf.nn.embedding_lookup(self.embedding, inputs)
        for i in range(self.num_layers):
            out, state = tf.nn.dynamic_rnn(self.cell, out, sequence_length=seq_length, initial_state=state, dtype=tf.float32)
        return out, state

class Decoder:
    def __init__(self, embedding, hidden_size, num_layers=1, max_length=15):
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = rnn.GRUCell(hidden_size)
        embedding_shape = tf.shape(embedding)
        self.linear = tf.Variable(tf.random_normal(shape=(self.hidden_size, embedding_shape[0]))*0.1)
        
        
    def __call__(self, inputs, state, encoder_state): # context vector
        
        out = tf.nn.embedding_lookup(self.embedding, inputs)
        out = tf.tile(tf.expand_dims(encoder_state, 1), (1, tf.shape(out)[1], 1))

        for i in range(self.num_layers):
#             state = tf.concat([state, encoder_state], 1)
            out, state = tf.nn.dynamic_rnn(self.cell, out, initial_state=state, dtype=tf.float32)
    
        out = tf.tensordot(out, self.linear, axes=[[2], [0]])
        return out, state

class Seq2Seq:
    def __init__(self, hidden_size, num_layers, embeddings):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = 15
        self.grad_clip = 5.0
        
        with tf.name_scope("place_holder"):
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int64, name="encoder_inputs")
            self.encoder_length = tf.placeholder(shape=(None, ), dtype=tf.int64, name="encoder_length")
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int64, name="decoder_inputs")
            self.decoder_target = tf.placeholder(shape=(None, None), dtype=tf.int64, name="decoder_target")
            self.decoder_mask = tf.placeholder(shape=(None, None), dtype=tf.float32, name="decoder_mask")

        with tf.name_scope("embedding"):
            self.embeddings = tf.get_variable(name="embeddings", dtype=tf.float32, shape=(len(embeddings), 300),
                                                initializer=tf.constant_initializer(embeddings), trainable=False)

        with tf.name_scope("encoder-decoder"):
            self.encoder = Encoder(self.embeddings, self.hidden_size, self.num_layers)
            self.decoder = Decoder(self.embeddings, self.hidden_size, self.num_layers)

        with tf.variable_scope("seq2seq-train"):
            encoder_outputs, encoder_state = self.encoder(self.encoder_inputs, self.encoder_length)
            tf.get_variable_scope().reuse_variables()
            # 这里把encoder的state送给decoder的state，encoder的output则不需要用了……
            decoder_state = encoder_state
            word_indices = self.decoder_inputs

            decoder_outputs, decoder_state = self.decoder(word_indices, decoder_state, encoder_state)

            # decoder_outputs.append(decoder_out)
            decoder_outputs = tf.concat(decoder_outputs, 1)

        with tf.name_scope("cost"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_outputs, labels=self.decoder_target)

            self.cost = tf.reduce_mean(loss * self.decoder_mask)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        with tf.variable_scope("seq2seq-generate"):
            self.generate_outputs = []
            decoder_state = encoder_state
            word_indices = tf.expand_dims(self.decoder_inputs[:, 0], 1)
            for i in range(self.max_length):
                decoder_out, decoder_state = self.decoder(word_indices, decoder_state, encoder_state)
                softmax_out = tf.nn.softmax(decoder_out[:, 0, :])
                word_indices = tf.expand_dims(tf.cast(tf.argmax(softmax_out, -1), dtype=tf.int64), 1)
                self.generate_outputs.append(word_indices)
            self.generate_outputs = tf.concat(self.generate_outputs, 0)
            
            
    def train(self, sess, encoder_inputs, encoder_length, decoder_inputs, decoder_target, decoder_mask):
        _, cost = sess.run([self.train_op, self.cost], feed_dict={
            self.encoder_inputs: encoder_inputs, 
            self.encoder_length: encoder_length,
            self.decoder_inputs: decoder_inputs,
            self.decoder_target: decoder_target,
            self.decoder_mask: decoder_mask
        })
        return cost
    
    def generate(self, sess, encoder_inputs, encoder_length):
        answer = [1]
        for i in range(15):
            print(answer)
            decoder_inputs = np.asarray([answer], dtype="int64")
            generate = sess.run([self.generate_outputs],
                               feed_dict={self.encoder_inputs: encoder_inputs,
                                          self.decoder_inputs: decoder_inputs,
                                          self.encoder_length: encoder_length})[0]
            print(generate[i-1])
            answer.extend(generate[i-1])
            print(answer)
        return generate
