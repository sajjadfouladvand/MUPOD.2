from __future__ import absolute_import, division, print_function, unicode_literals

#try:
#  %tensorflow_version 2.x
#except Exception:
#  pass
#import tensorflow_datasets as tfds
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#from sklearn import metrics

import time
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import os
import math
import scipy.special as sc
import random
from sklearn import metrics
from keras import regularizers

# min_epoch = 10
F1_idx = 10
#================= Read test performances and find the best model =================
testing_filename_meds_diags_procs_demogs='outputs/test_meds_diags_procs_demogs_represented.csv'

pdb.set_trace()
validation_res=[]
with open("results/single_stream_transformer/validation_results_single_stream_tranformer.csv") as validation_results:
  next(validation_results)
  for line in validation_results:
      line_perf=line.replace(',\n','').split(',')
      #pdb.set_trace()
      # line_perf[17].replace('\n','')
      line_perf = [float(i) for i in line_perf]  
      validation_res.append(line_perf)
pdb.set_trace()
validation_res_ar=np.array(validation_res)
best_validation_res_ar=validation_res_ar[np.argmax(validation_res_ar[:,F1_idx]),:]
best_model_number= int(best_validation_res_ar[0])
learning_rate= best_validation_res_ar[1]
num_layers=int(best_validation_res_ar[2])
num_heads = int(best_validation_res_ar[3])
BATCH_SIZE=int(best_validation_res_ar[4])
dropout_rate=best_validation_res_ar[5]
EPOCHS=int(best_validation_res_ar[6])
optimum_threshold=best_validation_res_ar[17]
#===================================================
#===================================================
model_number = 3000

#============== Read best model validation performances for early stopping
# pdb.set_trace()
# with open("validation_iterative_performance_thresholding_"+str(best_model_number)+".csv") as val_itr_perf_file:
#   next(val_itr_perf_file)
#   F1_scores_itr = []
#   for line in val_itr_perf_file:
#       F1_scores_itr.append(float(line.split(',')[3]))
# # pdb.set_trace()
# min_loss_idx = np.array(F1_scores_itr).argsort()[:EPOCHS]
# # pdb.set_trace()
# optimum_epoch = min_loss_idx[np.where(min_loss_idx >= min_epoch)[0][-1]] + 1
# pdb.set_trace()

# #============== Read validation losses and perform early stopping
# # pdb.set_trace()
# best_model_validation_loss = []
# with open("validation_loss_twostreamT_thresholding"+str(best_model_number)+".csv") as val_loss_file:
#   for line in val_loss_file:
#       best_model_validation_loss.append(float(line))
# # pdb.set_trace()
# min_loss_idx = np.array(best_model_validation_loss).argsort()[:EPOCHS]
# # pdb.set_trace()
# optimum_epoch = min_loss_idx[np.where(min_loss_idx >= min_epoch)[0][0]]

#import matplotlib.pyplot as plt
#CUDA_VISIBLE_DEVICES=3
#tf.enable_eager_execution() 
# pdb.set_trace()
#tf.compat.v1.enable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.keras.backend.clear_session()
random.seed(time.clock())

num_time_steps=138
d_input = 32
one=1
zero=0
num_classes=2
d_model = 32#128
dff = 32#512
input_vocab_size =  8500#(2 ** (d_meds + d_diags)) - 1 # tokenizer_pt.vocab_size + 2
target_vocab_size = 8000#(2 ** 2) - 1  #tokenizer_en.vocab_size + 2
# num_heads = 1#8
# num_thresholds=100
# learning_rate_pool =[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]#[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.08, 0.1]
# EPOCHS_pool=[20, 40, 80, 120, 150]#10, 15, 20]#[10000, 100000, 1000000, 2000000, 5000000, 10000000, 15000000]
# num_layers_pool=[1, 2, 4, 8]#, 16]#[10, 20, 40, 60, 80, 100, 140, 180, 200, 300, 600]
# BATCH_SIZE_pool=[64, 128, 256, 512]
# dropout_rate_pool=[0.3,  0.4,  0.5,  0.6]
# num_heads_pool = [1,2,4,8]
# # early_stopping_threshold_pool=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

# learning_rate= random.choice(learning_rate_pool)
# num_layers= random.choice(num_layers_pool)
# BATCH_SIZE=random.choice(BATCH_SIZE_pool)
# dropout_rate=random.choice(dropout_rate_pool)
# EPOCHS= random.choice(EPOCHS_pool)
# num_heads = random.choice(num_heads_pool)
# early_stopping_threshold=random.choice(early_stopping_threshold_pool)
def find_divisables(n):
  divisables = []
  for i in range(n):
    if n%(i+1) == 0:
      divisables.append(i+1)
  return divisables    

class ReadingData(object):
    def __init__(self, path_t="", path_l=""):#, path_s=""):
        
        self.data = []
        s=[]
        with open(path_t) as f:
              counter=0
              for line in f:
                  # counter+=1
                  # if counter>2000:
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
  #pdb.set_trace()
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
    #pdb.set_trace()
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
    self.final_layer = tf.keras.layers.Dense(num_classes)
    
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

# temp_learning_rate_schedule = CustomSchedule(d_model)


def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  #loss_ = loss_object(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  #pdb.set_trace()
  loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real) # + tf.nn.l2_loss(pred)
  # manual_cost=tf.reduce_mean(-tf.math.reduce_sum(tf.math.multiply(real, tf.math.log(tf.nn.softmax(pred) + 1e-10)), 1) + (1-tf.math.abs(tf.nn.softmax(pred)[:,0]-tf.nn.softmax(pred)[:,1]))) 

  return tf.reduce_mean(loss)#tf.reduce_mean(loss)

transformer = Transformer(num_layers, d_model, num_heads, dff, 
  input_vocab_size, target_vocab_size, dropout_rate)


# def create_masks(medications, tar):
#   # Encoder padding mask
#   enc_padding_mask = create_padding_mask(medications)
  
#   # Used in the 2nd attention block in the decoder.
#   # This padding mask is used to mask the encoder outputs.
#   dec_padding_mask = create_padding_mask(medications)
  
#   # Used in the 1st attention block in the decoder.
#   # It is used to pad and mask future tokens in the input received by 
#   # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#   dec_target_padding_mask = create_padding_mask(tar)
#   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
#   return enc_padding_mask, combined_mask, dec_padding_mask

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
#===========================SAVING MODELS==================
# checkpoint_path = "./checkpoints/train"
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')
#==========================================================
# train_step_signature = [
#     tf.TensorSpec(shape=(BATCH_SIZE, num_time_steps, d_meds), dtype=np.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, num_time_steps, d_diags), dtype=np.float32),
#     tf.TensorSpec(shape=(BATCH_SIZE, num_classes), dtype=np.float32),
# ]

# @tf.function(input_signature=train_step_signature)
# def train_step(medications, diagnoses, demog_info, tar):
#   #pdb.set_trace()
#   #tar_inp = tar#[:, :-1]
#   #tar_real = tar#[:, 1:]
#   #enc_padding_mask=tf.to_float(tf.random.uniform((84,1,1,12), minval=0, maxval=1, dtype=tf.float32))


# ======= Restoring the model
pdb.set_trace()
checkpoint_path =  "saved_models/checkpoints_single_stream/trained_model_" +str(best_model_number) 
checkpoint = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
# c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=optimum_epoch)#, ...)
c_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)#, ...)

if c_manager.latest_checkpoint:
    tf.print("-----------Restoring from {}-----------".format(
        c_manager.latest_checkpoint))
    checkpoint.restore(c_manager.latest_checkpoint)
    # optimum_epoch = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]

# if optimum_epoch == '':
#     if c_manager.latest_checkpoint:
#         tf.print("-----------Restoring from {}-----------".format(
#             c_manager.latest_checkpoint))
#         checkpoint.restore(c_manager.latest_checkpoint)
#         optimum_epoch = c_manager.latest_checkpoint.split(sep='ckpt-')[-1]
#     else:
#         tf.print("-----------Initializing from scratch-----------")
# else:    
#     checkpoint_fname = checkpoint_path + '/'+ 'ckpt-' + str(optimum_epoch)
#     tf.print("-----------Restoring from {}-----------".format(checkpoint_fname))
#     checkpoint.restore(checkpoint_fname)


# model_path =  "./checkpoints/trained_model_" +str(best_model_number) + '_' + str(optimum_epoch) + '/'
# transformer.restoreVariables()
# checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) #+ '_' + str(optimum_epoch)
# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)                           
# if checkpoint_path+'/ckpt-'+str(optimum_epoch):
#   ckpt.restore(checkpoint_path+'/ckpt-'+str(optimum_epoch)).assert_consumed()
#   #ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()
# else:
#   pdb.set_trace()
#   print("Error: checkpoint wasn't found!")


# checkpoint_path =  "./checkpoints/trained_model_" +str(best_model_number) + '_' + str(optimum_epoch)
# ckpt = tf.train.Checkpoint(transformer=transformer)#, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint).assert_consumed()
#   print ('Latest checkpoint restored-TWO STREAM NODEL!!')
#   print('MOdel number is: ', best_model_number)
#   print('Early stopping epoch is, ', optimum_epoch)

# testing_labels_filename='testing_labels_shuffled_balanced.csv'
testset = ReadingData(path_t=testing_filename_meds_diags_procs_demogs)#, path_l=testing_labels_filename)

testing_data = testset.data
# testing_label = testset.labels
testing_data_ar=np.array(testing_data)
testing_enrolids = testing_data_ar[:,0]
testing_set=np.reshape(testing_data_ar[:,one:-5],(len(testing_data_ar), num_time_steps, d_input))   
testset_labels= testing_data_ar[:,-5:-3]#np.array(testing_label)

test_split_k_all = find_divisables(len(testing_set))
testing_split_k = int(len(testing_set)/test_split_k_all[1])
#========================================
#pdb.set_trace()
# sess = tf.Session()
# variables_names = [v.name for v in transformer.trainable_variables()]
# values = sess.run(variables_names)
# for k, v in zip(variables_names, values):
#     print("Variable: ", k)
#     print("Shape: ", v.shape)
#     print(v)
# early_stopping=False


#=========================== Testing
#===== Breaking test set to k sets
testing_set_split=tf.split(value=testing_set, num_or_size_splits=testing_split_k)
logits_testing_all = list()
# logits_validation_all = np.zeros((len(medications_validation), num_classes))
for i in range(len(testing_set_split)):
  val_enc_mask = create_padding_mask(testing_set_split[i])
  logits_testing = transformer(testing_set_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
  logits_testing_all.extend(logits_testing)
predictions_testing_soft=tf.nn.softmax(logits_testing_all)
np.savetxt('results/single_stream_transformer/restoring_best_singlestreamT_model_testing_enrolids.csv', testing_enrolids, delimiter=',')
np.savetxt('results/single_stream_transformer/restoring_best_signlestreamT_model_predictions_testing_soft'+'_'+str(model_number)+'.csv', predictions_testing_soft)
np.savetxt('results/single_stream_transformer/restoring_best_singlestreamT_model_logits_testing'+'_'+str(model_number)+'.csv', logits_testing_all)
# pdb.set_trace() 


#============ Thresholding for test=================
probabilities_test_pos=predictions_testing_soft.numpy()[:,0]
# current_threshold_temp=0
# thresholding_results=np.zeros((num_thresholds, 11)) 
# y = np.array([1, 1, 2, 2])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# pdb.set_trace()
fpr, tpr, thresholds = metrics.roc_auc_score(testset_labels[:,0], probabilities_test_pos)
test_auc = metrics.auc(fpr, tpr)
# rf_test_auc=roc_auc_score(testset_labels[:,0], best_rf_model.predict_proba(test_data_ar[:,1:-2])[:,1])
tp_test=0
tn_test=0
fp_test=0
fn_test=0
for i in range(len(probabilities_test_pos)):      
    if(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==0):
        tn_test=tn_test+1
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==1):
        tp_test=tp_test+1
    elif(probabilities_test_pos[i]>=optimum_threshold and testset_labels[i,0]==0):
        fp_test=fp_test+1
    elif(probabilities_test_pos[i]<optimum_threshold and testset_labels[i,0]==1):
        fn_test=fn_test+1
if((tp_test+fp_test)==0):
    precision_test=0
else:
    precision_test=tp_test/(tp_test+fp_test)
recall_test=tp_test/(tp_test+fn_test)
sensitivity_test=tp_test/(tp_test+fn_test)
specificity_test=tn_test/(tn_test+fp_test)    
if (precision_test+recall_test) !=0:
    F1Score_test=(2*precision_test*recall_test)/(precision_test+recall_test)      
else:
    F1Score_test=0        
accuracy_test= (tp_test+tn_test)/(tp_test+tn_test+fp_test+fn_test)
# pdb.set_trace()

header_results_filename= "Model Number (SINGLE STREAM), Learning Rate, Number of Layeres, Batch Size, Dropout Rate, Number of EPOCHS, test Accuracy, test Precision, test Recall, test F1-Score, test Specificity, test TP, test TN , test FP, test FN, test auc, test optimum threshold \n"
print("=================== SINGLE STREAM MODEL======================")
#pdb.set_trace()
with open("results/single_stream_transformer/restoring_best_singlestreamT_model_test_results__thresholdin.csv", 'w') as results_file:
      results_file.write("".join(["".join(x) for x in header_results_filename]))  
      results_file.write(str(best_model_number))
      results_file.write(",")
      results_file.write(str(learning_rate))
      results_file.write(",")
      results_file.write(str(num_layers))
      results_file.write(",")
      results_file.write(str(BATCH_SIZE))
      results_file.write(",")
      results_file.write(str(dropout_rate))
      results_file.write(",")
      results_file.write(str(EPOCHS))
      results_file.write(",")
      results_file.write(str(accuracy_test))
      results_file.write(", ")
      results_file.write(str(precision_test))
      results_file.write(", ")
      results_file.write(str(recall_test))
      results_file.write(", ")
      results_file.write(str(F1Score_test))
      results_file.write(", ")
      results_file.write(str(specificity_test))
      results_file.write(", ")                     
      results_file.write(str(tp_test))
      results_file.write(", ")
      results_file.write(str(tn_test))
      results_file.write(", ")
      results_file.write(str(fp_test))
      results_file.write(", ")
      results_file.write(str(fn_test))
      results_file.write(",")
      results_file.write(str(test_auc))
      results_file.write(",")
      results_file.write(str(optimum_threshold))
      # results_file.write(",")
      # results_file.write(str(optimum_epoch))
      # results_file.write(",")  
      # results_file.write(str(regu_factor))
      results_file.write("\n")  
