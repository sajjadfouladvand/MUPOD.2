from __future__ import absolute_import, division, print_function, unicode_literals

#try:
#  %tensorflow_version 2.x
#except Exception:
#  pass
# import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import pdb
import os
import random
from sklearn import metrics
from keras import regularizers

#import matplotlib.pyplot as plt

#tf.enable_eager_execution() 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
tf.keras.backend.clear_session()
random.seed(time.clock())

num_time_steps=138
d_input = 48
one=1
zero=0
num_classes=2
d_model = 48#128
dff = 48#512
input_vocab_size =  8500#(2 ** (d_meds + d_diags)) - 1 # tokenizer_pt.vocab_size + 2
target_vocab_size = 8000#(2 ** 2) - 1  #tokenizer_en.vocab_size + 2
# num_heads = 1#8
num_thresholds=1000
# learning_rate_pool =[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]#[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.08, 0.1]
# EPOCHS_pool=[20, 40, 80, 120, 150]#10, 15, 20]#[10000, 100000, 1000000, 2000000, 5000000, 10000000, 15000000]
# num_layers_pool=[1, 2, 4, 8]#, 16]#[10, 20, 40, 60, 80, 100, 140, 180, 200, 300, 600]
# BATCH_SIZE_pool=[64, 128, 256, 512]
# dropout_rate_pool=[0.3,  0.4,  0.5,  0.6]
# num_heads_pool = [1,2,4,8]
# early_stopping_threshold_pool=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

learning_rate = 0.001#random.choice(learning_rate_pool)
num_layers= 4#random.choice(num_layers_pool)
BATCH_SIZE=256#random.choice(BATCH_SIZE_pool)
dropout_rate=0.3#random.choice(dropout_rate_pool)
EPOCHS= 5#random.choice(EPOCHS_pool)
regu_factor = 0.000001#random.choice(regularization_factor_pool)
num_heads = 4#random.choice(num_heads_pool)
# early_stopping_threshold=random.choice(early_stopping_threshold_pool)

class ReadingData(object):
    def __init__(self, path_t="", path_l=""):#, path_s=""):
        
        self.data = []
        s=[]
        with open(path_t) as f:
              # counter=0
              for line in f:
                  # counter+=1
                  # if counter>5000:
                  #   print('======================================WARNING=============')
                  #   print('You are reading topk samples.')
                  #   break
                  d_temp=line.split(',')
                  d_temp=[float(x) for x in d_temp]
                  self.data.append(d_temp)
                  d_temp=[]
                  s=[]
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data#, batch_labels#, batch_seqlen

def find_divisables(n):
  divisables = []
  for i in range(n):
    if n%(i+1) == 0:
      divisables.append(i+1)
  return divisables    
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  #pdb.set_trace()
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(inp):
  # pdb.set_trace()
  #======== Find which stream has more records
  # np.sum(meds,axis=2)
  # with open('error_log','a') as err_file:
  #   err_file.write("Creating masks...\n")  
  seq=np.ones((inp.shape[0], num_time_steps, d_model))
  for i in range(inp.shape[0]):
    # print(i)
    # if i==32:
    # pdb.set_trace()
    inp_non_zero_map = np.equal(inp[i],0).all(axis=1)
    # diags_non_zero_map = np.equal(diags[i],0).all(axis=1)
    if np.where(inp_non_zero_map==True)[0].size !=0:
      inp_zero_start = np.where(inp_non_zero_map==True)[0][0]
    else:
      inp_zero_start = num_time_steps-1
    # if np.where(diags_non_zero_map==True)[0].size != 0:
    #   diags_zero_start = np.where(diags_non_zero_map==True)[0][0]
    # else:
    #   diags_zero_start =  num_time_steps-1
    seq[i][inp_zero_start:]=0
  # pdb.set_trace()
  seq=tf.math.reduce_sum(seq, axis=2)
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]   

  # pdb.set_trace()
  # seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # # add extra dimensions to add the padding
  # # to the attention logits.
  # return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  # pdb.set_trace()
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  if mask is not None:
    attention_weights *= tf.transpose((1-mask), perm=[0,1,3,2])

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    #pdb.set_trace()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense((d_model)*num_time_steps)
    self.wk = tf.keras.layers.Dense((d_model)*num_time_steps)
    self.wv = tf.keras.layers.Dense((d_model)*num_time_steps)
    
    self.dense = tf.keras.layers.Dense(d_model*num_time_steps)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    #pdb.set_trace()
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    # pdb.set_trace()
    batch_size = tf.shape(q)[0]
    q_reshaped=tf.reshape(q, (batch_size, -1))
    k_reshaped=tf.reshape(k, (batch_size, -1))
    v_reshaped=tf.reshape(v, (batch_size, -1))
    #q = tf.reshape(self.wq(tf.reshape(q, (batch_size, -1))), (batch_size, num_time_steps, (d_meds + d_diags)))
    q = self.wq(q_reshaped)  # (batch_size, seq_len, d_model)
    k = self.wk(k_reshaped)  # (batch_size, seq_len, d_model)
    v = self.wv(v_reshaped)  # (batch_size, seq_len, d_model)
    
    q= tf.reshape(q, (batch_size, num_time_steps, (d_model)))
    k= tf.reshape(k, (batch_size, num_time_steps, (d_model)))
    v= tf.reshape(v, (batch_size, num_time_steps, (d_model)))

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
    # pdb.set_trace()
    concat_attention_reshaped = tf.reshape(concat_attention, (batch_size, -1))
    output = self.dense(concat_attention_reshaped)  # (batch_size, seq_len_q, d_model)
    output = tf.reshape(output, (batch_size, num_time_steps, d_model))
    #======= I added the following line to avoid the dense layer error
    #output = concat_attention       
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
# pdb.set_trace()
# sample_ffn = point_wise_feed_forward_network(512, 2048)
# sample_ffn(tf.random.uniform((64, 50, 512))).shape
#pdb.set_trace()
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    #pdb.set_trace()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    #Sajjad: this function sends the input matrix x to the MultiheadAttention to calculate attention weights and
    #outputs (using equation 1 in the attention is all you need). Then pass the output of the
    #MultiheadAttention to a drouput and a layer normalization and a ffn (a feed forward neural network) and
    # another dropout and normalization. In fact, it performs the encoder part (left part) in 
    # Figure 1 in "Attention is all you need paper". In the encoder part of Figure 1 from buttom
    #to top is: MultiheadAttention, Add & Norm, Feed Forward, Add & Norm    
    # pdb.set_trace()
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    #pdb.set_trace()          
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()
    #pdb.set_trace()
    self.d_model = d_model
    self.num_layers = num_layers
    
    #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):
    # pdb.set_trace()
    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
      #pdb.set_trace()
      #print("test the dimension")
    return x  # (batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()
    #pdb.set_trace()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    #self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
    #                       target_vocab_size, rate)
    self.input_map = tf.keras.layers.Dense(d_model)
    self.final_layer = tf.keras.layers.Dense(num_classes, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))

  def call(self, inp, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    # pdb.set_trace()
    batch_size = tf.shape(inp)[0]
    inp = self.input_map(inp)
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    # pdb.set_trace()
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    #dec_output, attention_weights = self.decoder(
    #    tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    enc_output_reshaped=tf.reshape(enc_output,(batch_size, -1))
    final_output = self.final_layer(enc_output_reshaped)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output#, attention_weights


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()
#     #pdb.set_trace()
#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps
    
#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)
#     #pdb.set_trace()
#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
#pdb.set_trace()
#learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
# optimizer = tf.keras.optimizers.SGD(learning_rate)
# temp_learning_rate_schedule = CustomSchedule(d_model)


def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  #loss_ = loss_object(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  #pdb.set_trace()
  # loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=real)#, pos_weight=0.00001) #+ tf.nn.l2_loss(pred)
  loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)#, pos_weight=0.00001) #+ tf.nn.l2_loss(pred)
  # manual_cost=tf.reduce_mean(-tf.math.reduce_sum(tf.math.multiply(real, tf.math.log(tf.nn.softmax(pred) + 1e-10)), 1) + (1-tf.math.abs(tf.nn.softmax(pred)[:,0]-tf.nn.softmax(pred)[:,1]))) 

  return tf.reduce_mean(loss)#tf.reduce_mean(loss)

transformer = Transformer(num_layers, d_model, num_heads, dff, 
  input_vocab_size, target_vocab_size, dropout_rate)


# checkpoint_path = "./checkpoints/train"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')



# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# train_step_signature = [
#     tf.TensorSpec(shape=(BATCH_SIZE, num_time_steps, (d_meds + d_diags)), dtype=tf.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, num_classes), dtype=tf.float32),
# ]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  # pdb.set_trace()
  #tar_inp = tar[:, :-1]
  #tar_real = tar[:, 1:]
  
  #enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  #enc_padding_mask=None
  #combined_mask= None
  #dec_padding_mask=None
  enc_padding_mask = create_padding_mask(inp)

  with tf.GradientTape() as tape:
    logits = transformer(inp, True, 
                              enc_padding_mask=enc_padding_mask, 
                              look_ahead_mask=None, 
                              dec_padding_mask=None)
    #pdb.set_trace()
    loss = loss_function(tar, logits)
  #pdb.set_trace()  
  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  #train_loss(loss)
  #train_accuracy(tar, predictions)
  predictions=tf.nn.softmax(logits)
  return predictions, loss, logits
#pdb.set_trace() 

#=====================================================
#=====================================================
#==============Reading data===========================


validation_filename_meds_diags_demogs='outputs/validation_meds_diags_procs_demogs_represented.csv'
# validation_labels_filename='validation_labels_shuffled_balanced.csv'
validationset = ReadingData(path_t=validation_filename_meds_diags_demogs)#, path_l=validation_labels_filename)

validation_data = validationset.data
# validation_label = validationset.labels
validation_data_ar=np.array(validation_data)
validation_enrolids = validation_data_ar[:,0]
validation_set=np.reshape(validation_data_ar[:,one:-5],(len(validation_data_ar), num_time_steps, d_input))   
validationset_labels=validation_data_ar[:,-5:-3]

validation_split_k_all = find_divisables(len(validation_set))
validation_split_k = int(len(validation_set)/validation_split_k_all[1])

train_filename='outputs/train_meds_diags_procs_demogs_represented.csv'
# train_labels_filename='training_labels_shuffled_balanced.csv'
trainset_meds = ReadingData(path_t=train_filename)#, path_l=train_labels_filename)
# pdb.set_trace() 
train_set_shape = np.shape(trainset_meds.data)
real_labels=np.zeros((BATCH_SIZE, num_classes), dtype=np.float32)

#=====================================================
#=====================================================
#model_number=0
random.seed(time.clock())
model_number=1234#random.randint(1, 1000000)
print("TRAINED MODEL NUMBER: ", model_number)
print("Number of epochs is: ,", EPOCHS)
# pdb.set_trace()
with open("results/single_stream_transformer/training_loss_singleStream_transformer_"+str(model_number)+".csv", 'w') as loss_file, open("results/single_stream_transformer/validation_loss_singleStream_transformer_"+str(model_number)+".csv", 'w') as val_loss_file:
  for epoch in range(EPOCHS):
    start = time.time()
    print("============= Epoch number========", epoch)
    # pdb.set_trace()
    #train_loss.reset_states()
    #train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    #for (batch, (inp, tar)) in enumerate(train_dataset):
    step=1
    while step * BATCH_SIZE < train_set_shape[0]:    
      #
      #==============================================
      #==============================================
      #================ Reading batches from training data===========
      batch_x = trainset_meds.next(BATCH_SIZE)
      # batch_y_ar=np.array(batch_y)
      # real_labels=
      # batch_y_ar[:,one:].astype(np.float32)
      batch_x_ar=np.array(batch_x)
      real_labels = batch_x_ar[:,-5:-3]
      batch_x_ar_reshaped=np.reshape(batch_x_ar[:,one:-5],(BATCH_SIZE, num_time_steps, d_input))   
      meds_diags_demogs_rep = tf.convert_to_tensor(batch_x_ar_reshaped, np.float32)
      real_labels=tf.convert_to_tensor(real_labels, np.float32)

      predictions, loss, logits = train_step(meds_diags_demogs_rep, real_labels)
      loss_file.write(str(loss.numpy()))
      loss_file.write('\n')       
      step = step + 1
    #==================================================================
    #==================================================================
    #==================Early stopping==================================
    if epoch % 5 ==0:
      # pdb.set_trace()
      validation_set_split=tf.split(value=validation_set, num_or_size_splits=validation_split_k)
      logits_validation_all = []
      # logits_validation_all = np.zeros((len(medications_validation), num_classes))
      for i in range(len(validation_set_split)):
        val_enc_mask = create_padding_mask(validation_set_split[i])
        
        logits_validation = transformer(validation_set_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
        logits_validation_all.extend(logits_validation)
      # pdb.set_trace()              
      # val_enc_mask = create_padding_mask(medications_validation, diagnoses_validation)
      # logits_validation = transformer(medications_validation, diagnoses_validation, validation_demog_info, False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
      val_loss= loss_function(validationset_labels, logits_validation_all)
      val_loss_file.write(str(val_loss.numpy()))
      val_loss_file.write("\n")
   
# pdb.set_trace()   
checkpoint_path = "saved_models/checkpoints_single_stream/trained_model_" + str(model_number)
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
ckpt_save_path = ckpt_manager.save()


validation_set_split=tf.split(value=validation_set, num_or_size_splits=validation_split_k)
logits_validation_all = []
# logits_validation_all = np.zeros((len(medications_validation), num_classes))
for i in range(len(validation_set_split)):
  val_enc_mask = create_padding_mask(validation_set_split[i])
  
  logits_validation = transformer(validation_set_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
  logits_validation_all.extend(logits_validation)
predictions_validation_soft=tf.nn.softmax(logits_validation_all)
np.savetxt('results/single_stream_transformer/validation_enrolids_single_stream.csv', validation_enrolids, delimiter=',')
np.savetxt('results/single_stream_transformer/predictions_validation_soft'+'_'+str(model_number)+'.csv', predictions_validation_soft)
np.savetxt('results/single_stream_transformer/logits_validation'+'_'+str(model_number)+'.csv', logits_validation)
# pdb.set_trace() 
#============ Thresholding for Validation=================
probabilities_validation_pos=predictions_validation_soft.numpy()[:,0]
current_threshold_temp=0
thresholding_results=np.zeros((num_thresholds, 11)) 
# y = np.array([1, 1, 2, 2])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# pdb.set_trace()
# fpr, tpr, thresholds = metrics.roc_curve(validationset_labels[:,0], probabilities_validation_pos, pos_label=1)
# validation_auc = metrics.auc(fpr, tpr)
validation_auc=metrics.roc_auc_score(validationset_labels[:,0], probabilities_validation_pos)  
# rf_test_auc=roc_auc_score(validationset_labels[:,0], best_rf_model.predict_proba(test_data_ar[:,1:-2])[:,1])
for thresh_index in range(num_thresholds):
    # current_threshold_temp= current_threshold_temp + 0.001              
    current_threshold_temp = current_threshold_temp + (1/num_thresholds)              
    tp_validation=0
    tn_validation=0
    fp_validation=0
    fn_validation=0
    for i in range(len(probabilities_validation_pos)):      
        if(probabilities_validation_pos[i]<current_threshold_temp and validationset_labels[i,0]==0):
            tn_validation=tn_validation+1
        elif(probabilities_validation_pos[i]>=current_threshold_temp and validationset_labels[i,0]==1):
            tp_validation=tp_validation+1
        elif(probabilities_validation_pos[i]>=current_threshold_temp and validationset_labels[i,0]==0):
            fp_validation=fp_validation+1
        elif(probabilities_validation_pos[i]<current_threshold_temp and validationset_labels[i,0]==1):
            fn_validation=fn_validation+1
    if((tp_validation+fp_validation)==0):
        precision_validation=0
    else:
        precision_validation=tp_validation/(tp_validation+fp_validation)
    recall_validation=tp_validation/(tp_validation+fn_validation)
    sensitivity_validation=tp_validation/(tp_validation+fn_validation)
    specificity_validation=tn_validation/(tn_validation+fp_validation)    
    if (precision_validation+recall_validation) !=0:
        F1Score_validation=(2*precision_validation*recall_validation)/(precision_validation+recall_validation)      
    else:
        F1Score_validation=0        
    accuracy_validation= (tp_validation+tn_validation)/(tp_validation+tn_validation+fp_validation+fn_validation)
    thresholding_results[thresh_index, 0] = tp_validation
    thresholding_results[thresh_index, 1] = tn_validation
    thresholding_results[thresh_index, 2] = fp_validation
    thresholding_results[thresh_index, 3] = fn_validation
    thresholding_results[thresh_index, 4] = accuracy_validation
    thresholding_results[thresh_index, 5] = specificity_validation
    thresholding_results[thresh_index, 6] = precision_validation
    thresholding_results[thresh_index, 7] = recall_validation
    thresholding_results[thresh_index, 8] = F1Score_validation
    thresholding_results[thresh_index, 9] = current_threshold_temp
    thresholding_results[thresh_index, 10] = validation_auc
# pdb.set_trace()
best_validation_results=thresholding_results[np.argmax(thresholding_results[:,8]),:]
tp_validation=best_validation_results[0]
tn_validation=best_validation_results[1]
fp_validation=best_validation_results[2]
fn_validation=best_validation_results[3]
accuracy_validation=best_validation_results[4]
specificity_validation=best_validation_results[5]
precision_validation=best_validation_results[6]
recall_validation=best_validation_results[7]
F1Score_validation=best_validation_results[8]
optimum_threshold=best_validation_results[9] 
validation_auc = best_validation_results[10] 


print("Model is saved, the single stream model number is: ", model_number)
header_results_filename= "Model Number (SINGLE STREAM), Learning Rate, Number of Layeres, Number of heads, Batch Size, Dropout Rate, Number of EPOCHS, Validation Accuracy, Validation Precision, Validation Recall, Validation F1-Score, Validation Specificity, Validation TP, Validation TN , Validation FP, Validation FN, Valication AUC, validation optimum threshold, regularization\n"
print("=================== SINGLE STREAM MODEL======================")
#pdb.set_trace()
with open("results/single_stream_transformer/validation_results_single_stream_tranformer.csv", 'a') as results_file:
      results_file.write("".join(["".join(x) for x in header_results_filename]))  
      results_file.write(str(model_number))
      results_file.write(",")
      results_file.write(str(learning_rate))
      results_file.write(",")
      results_file.write(str(num_layers))
      results_file.write(",")
      results_file.write(str(num_heads))
      results_file.write(",")      
      results_file.write(str(BATCH_SIZE))
      results_file.write(",")
      results_file.write(str(dropout_rate))
      results_file.write(",")
      results_file.write(str(EPOCHS))
      results_file.write(",")
      results_file.write(str(accuracy_validation))
      results_file.write(", ")
      results_file.write(str(precision_validation))
      results_file.write(", ")
      results_file.write(str(recall_validation))
      results_file.write(", ")
      results_file.write(str(F1Score_validation))
      results_file.write(", ")
      results_file.write(str(specificity_validation))
      results_file.write(", ")                     
      results_file.write(str(tp_validation))
      results_file.write(", ")
      results_file.write(str(tn_validation))
      results_file.write(", ")
      results_file.write(str(fp_validation))
      results_file.write(", ")
      results_file.write(str(fn_validation))
      results_file.write(",")
      results_file.write(str(validation_auc))
      results_file.write(",")
      results_file.write(str(optimum_threshold))
      results_file.write(",")
      results_file.write(str(regu_factor))        
      results_file.write("\n")    



