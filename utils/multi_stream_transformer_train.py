from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import math
import scipy.special as sc
import random
from sklearn import metrics
from keras import regularizers

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
tf.keras.backend.clear_session()
random.seed(time.clock())

# === File names
validation_filename_meds='outputs/validation_meds_represented.csv'
validation_filename_diags='outputs/validation_diags_represented.csv'
validation_filename_procs='outputs/validation_procs_represented.csv'
validation_filename_demogs='outputs/validation_demographics_multihot.csv'

train_filename_meds='outputs/train_meds_represented.csv'
train_filename_diags='outputs/train_diags_represented.csv'
train_filename_procs = 'outputs/train_procs_represented.csv'
train_filename_demogs = 'outputs/train_demographics_multihot.csv'

print('WARNING: YOU ARE USING FIRST 100 SAMPLLES')
# n_temp =100
# ==== constants
d_model=48
dff = 48
input_vocab_size = 8500
target_vocab_size = 8000
d_meds=16
d_diags=16
d_procs=16
d_demogs=2
zero=0
num_time_steps = 138
num_classes=2
one=1
two=2
num_thresholds=1000

# ======================= Generating a set of model parameters randomely
learning_rate_pool =[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
EPOCHS_pool= [20, 40, 80, 120, 150]
num_layers_pool=[1, 2, 4, 8]
BATCH_SIZE_pool=[64, 128, 256, 512]
dropout_rate_pool=[0.3, 0.4, 0.5, 0.6]
regularization_factor_pool = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
num_heads_pool = [1,2,4,8]

learning_rate= 0.001#random.choice(learning_rate_pool)
num_layers=4#random.choice(num_layers_pool)
BATCH_SIZE=256#random.choice(BATCH_SIZE_pool)
dropout_rate=0.3#random.choice(dropout_rate_pool)
EPOCHS=5#random.choice(EPOCHS_pool)
regu_factor = 0.000001#random.choice(regularization_factor_pool)
num_heads = 4#random.choice(num_heads_pool)

# ========= A function to read the data streams
class ReadingData(object):
    def __init__(self, path_t="", path_l=""):
        """
        Reading the file in path_t. Path_l doesn't do anything and will be removed from next versions.
        """
        self.data = []
        self.labels = []
        self.seqlen = []
        s=[]
        temp=[]
        with open(path_t) as f:
              for line in f:                   
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
        return batch_data

def find_divisables(n):
  divisables = []
  for i in range(n):
    if n%(i+1) == 0:
      divisables.append(i+1)
  return divisables    

# ==== Original function from the Transformer code by Google. This code is part of the positional encoding mechanism
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

# ==== Original function from the Transformer code by Google. This code is part of the positional encoding mechanism
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


# ==== This fnction masks the information after the last visit of the patients
def create_padding_mask(meds, diags, procs): 
  """
  Inputs are medication and diagnoses streams 
  This function finds the maximum index where the features (for either medication or diagnoses) 
  are not zero for a visit i, and then creates a mask matrix to mask out everything after the last non-zero visit i.
  """
  seq=np.ones((meds.shape[0], num_time_steps, d_model))
  for i in range(meds.shape[0]):
    meds_non_zero_map = np.equal(meds[i],0).all(axis=1)
    diags_non_zero_map = np.equal(diags[i],0).all(axis=1)
    procs_non_zero_map = np.equal(procs[i],0).all(axis=1)
    if np.where(meds_non_zero_map==True)[0].size !=0:
      meds_zero_start = np.where(meds_non_zero_map==True)[0][0]
    else:
      meds_zero_start = num_time_steps-1
    if np.where(diags_non_zero_map==True)[0].size != 0:
      diags_zero_start = np.where(diags_non_zero_map==True)[0][0]
    else:
      diags_zero_start =  num_time_steps-1
    if np.where(procs_non_zero_map==True)[0].size != 0:
      procs_zero_start = np.where(procs_non_zero_map==True)[0][0]
    else:
      procs_zero_start =  num_time_steps-1      
    seq[i][max(meds_zero_start, diags_zero_start, procs_zero_start):]=0
  seq=tf.math.reduce_sum(seq, axis=2)
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  return seq[:, tf.newaxis, tf.newaxis, :]  


def scaled_dot_product_attention(v,k,q,mask):
  """Calculate the attention weights.  
  Args:
    q is query
    k is key
    v is value
    mask: created from create_padding_mask function. Since we have two streams, we mask out from the last non-zero 
    visit i in both directions (rows and columns)
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

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  if mask is not None:
    attention_weights *= tf.transpose((1-mask), perm=[0,1,3,2])
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wmq = tf.keras.layers.Dense(d_model)
    self.wmk = tf.keras.layers.Dense(d_model)
    self.wmv = tf.keras.layers.Dense(d_model)
    
    self.wdq = tf.keras.layers.Dense(d_model)
    self.wdk = tf.keras.layers.Dense(d_model)
    self.wdv = tf.keras.layers.Dense(d_model)

    self.wpq = tf.keras.layers.Dense(d_model)
    self.wpk = tf.keras.layers.Dense(d_model)
    self.wpv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, medications, diagnoses, procedures, mask):
    # pdb.set_trace()
    batch_size = tf.shape(medications)[0]

    # ===== Creating q, k, v for both medications and diagnoses streams using different dense layers 
    medications_q = self.wmq(medications)  
    medications_k = self.wmk(medications)  
    medications_v = self.wmv(medications)  

    diagnoses_q = self.wdq(diagnoses)  
    diagnoses_k = self.wdk(diagnoses)  
    diagnoses_v = self.wdv(diagnoses)

    procedures_q = self.wpq(procedures)  
    procedures_k = self.wpk(procedures)  
    procedures_v = self.wpv(procedures)
    
    medications_q = self.split_heads(medications_q, batch_size)  
    medications_k = self.split_heads(medications_k, batch_size)  
    medications_v = self.split_heads(medications_v, batch_size)  

    
    diagnoses_q = self.split_heads(diagnoses_q, batch_size)  
    diagnoses_k = self.split_heads(diagnoses_k, batch_size)  
    diagnoses_v = self.split_heads(diagnoses_v, batch_size) 

    procedures_q = self.split_heads(procedures_q, batch_size)  
    procedures_k = self.split_heads(procedures_k, batch_size)  
    procedures_v = self.split_heads(procedures_v, batch_size) 
    # pdb.set_trace()
   # ====== Calculating Z_MM: the attention weights and final outputs for Medications-Medications interaction. 
    scaled_attention_MM, attention_weights_MM = scaled_dot_product_attention(medications_v, medications_k, medications_q, mask)
    scaled_attention_MM = tf.transpose(scaled_attention_MM, perm=[0, 2, 1, 3])  
    concat_attention_MM = tf.reshape(scaled_attention_MM, 
                                  (batch_size, -1, self.d_model))  
    output_MM = self.dense(concat_attention_MM)  

  #======================= Calculating Z_MD: the attention weights and final outputs for Medication-Diagnoses interaction. 
    scaled_attention_MD, attention_weights_MD = scaled_dot_product_attention(diagnoses_v, diagnoses_k, medications_q, mask)
    scaled_attention_MD = tf.transpose(scaled_attention_MD, perm=[0, 2, 1, 3]) 
    concat_attention_MD = tf.reshape(scaled_attention_MD, 
                                  (batch_size, -1, self.d_model))  
    output_MD = self.dense(concat_attention_MD)  

  #======================= Calculating Z_MP: the attention weights and final outputs for Medication-Procedures interaction. 
    scaled_attention_MP, attention_weights_MP = scaled_dot_product_attention(procedures_v, procedures_k, medications_q, mask)
    scaled_attention_MP = tf.transpose(scaled_attention_MP, perm=[0, 2, 1, 3]) 
    concat_attention_MP = tf.reshape(scaled_attention_MP, 
                                  (batch_size, -1, self.d_model))  
    output_MP = self.dense(concat_attention_MP)

  #======================= Calculating Z_DD: the attention weights and final outputs for Diagnoses-Diagnoses interaction. 
    scaled_attention_DD, attention_weights_DD = scaled_dot_product_attention(diagnoses_v, diagnoses_k, diagnoses_q, mask)
    scaled_attention_DD = tf.transpose(scaled_attention_DD, perm=[0, 2, 1, 3]) 
    concat_attention_DD = tf.reshape(scaled_attention_DD, 
                                  (batch_size, -1, self.d_model))  
    output_DD = self.dense(concat_attention_DD)  

  #======================= Calculating Z_DP: the attention weights and final outputs for Diagnoses-Procedures interaction. 
    scaled_attention_DP, attention_weights_DP = scaled_dot_product_attention(procedures_v, procedures_k, diagnoses_q, mask)
    scaled_attention_DP = tf.transpose(scaled_attention_DP, perm=[0, 2, 1, 3]) 
    concat_attention_DP = tf.reshape(scaled_attention_DP, 
                                  (batch_size, -1, self.d_model))  
    output_DP = self.dense(concat_attention_DP)


  #======================= Calculating Z_PP: the attention weights and final outputs for Procedures-Procedures interaction. 
    scaled_attention_PP, attention_weights_PP = scaled_dot_product_attention(procedures_v, procedures_k, procedures_q, mask)
    scaled_attention_PP = tf.transpose(scaled_attention_PP, perm=[0, 2, 1, 3]) 
    concat_attention_PP = tf.reshape(scaled_attention_PP, 
                                  (batch_size, -1, self.d_model))  
    output_PP = self.dense(concat_attention_PP)  


    return output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_MP, attention_weights_MP, output_DD, attention_weights_DD, output_DP, attention_weights_DP, output_PP, attention_weights_PP

def point_wise_feed_forward_network(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  
  ])


def point_wise_feed_forward_network_M(d_model, dff):
  return  tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  
      tf.keras.layers.Dense(d_model)  
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
    self.M_hat = tf.keras.layers.Dense(num_time_steps * d_model)
    self.D_hat = tf.keras.layers.Dense(num_time_steps * d_model)
    self.P_hat = tf.keras.layers.Dense(num_time_steps * d_model)

  def call(self, medications, diagnoses, procedures, training, mask):
    # pdb.set_trace()
    batch_size = tf.shape(medications)[0]
    output_MM, attention_weights_MM, output_MD, attention_weights_MD, output_MP, attention_weights_MP, output_DD, attention_weights_DD, output_DP, attention_weights_DP, output_PP, attention_weights_PP = self.mha(medications,diagnoses, procedures, mask)  # (batch_size, input_seq_len, d_model)
    
    #============ dropout, normalization, feed forward, dropout and normalization. Inspired by the original idea in Transformers
    output_MM = self.dropout1(output_MM, training=training)
    output_MM_1 = self.layernorm1(medications + output_MM)   
    ffn_output_MM_1 = self.ffn(output_MM_1)  
    ffn_output_MM_1 = self.dropout2(ffn_output_MM_1, training=training)
    output_MM_2 = self.layernorm2(output_MM_1 + ffn_output_MM_1)  

    average_MD= tf.divide(tf.math.add_n([medications, diagnoses]), 2)
    output_MD = self.dropout1(output_MD, training=training)
    output_MD_1 = self.layernorm1(average_MD + output_MD)    
    ffn_output_MD_1 = self.ffn(output_MD_1)  
    ffn_output_MD_1 = self.dropout2(ffn_output_MD_1, training=training)
    output_MD_2 = self.layernorm2(output_MD_1 + ffn_output_MD_1) 

    average_MP= tf.divide(tf.math.add_n([medications, procedures]), 2)
    output_MP = self.dropout1(output_MP, training=training)
    output_MP_1 = self.layernorm1(average_MP + output_MP)    
    ffn_output_MP_1 = self.ffn(output_MP_1)  
    ffn_output_MP_1 = self.dropout2(ffn_output_MP_1, training=training)
    output_MP_2 = self.layernorm2(output_MP_1 + ffn_output_MP_1)     

    output_DD = self.dropout1(output_DD, training=training)
    output_DD_1 = self.layernorm1(diagnoses + output_DD)      
    ffn_output_DD_1 = self.ffn(output_DD_1)  
    ffn_output_DD_1 = self.dropout2(ffn_output_DD_1, training=training)
    output_DD_2 = self.layernorm2(output_DD_1 + ffn_output_DD_1)  

    average_DP= tf.divide(tf.math.add_n([diagnoses, procedures]), 2)
    output_DP = self.dropout1(output_DP, training=training)
    output_DP_1 = self.layernorm1(average_DP + output_DP)    
    ffn_output_DP_1 = self.ffn(output_DP_1)  
    ffn_output_DP_1 = self.dropout2(ffn_output_DP_1, training=training)
    output_DP_2 = self.layernorm2(output_DP_1 + ffn_output_DP_1)     
    
    output_PP = self.dropout1(output_PP, training=training)
    output_PP_1 = self.layernorm1(procedures + output_PP)      
    ffn_output_PP_1 = self.ffn(output_PP_1)  
    ffn_output_PP_1 = self.dropout2(ffn_output_PP_1, training=training)
    output_PP_2 = self.layernorm2(output_PP_1 + ffn_output_PP_1) 

    # ===== Concatenating all outputs related to the medication stream 
    concatenated_outputs_forM = tf.concat([output_MM_2, output_MD_2, output_MP_2], axis=2) 
    # ===== Concatenating all outputs related to the diagnoses stream             
    concatenated_outputs_forD = tf.concat([output_MD_2, output_DD_2, output_DP_2], axis=2)         

    # ===== Concatenating all outputs related to the diagnoses stream             
    concatenated_outputs_forP = tf.concat([output_MP_2, output_DP_2, output_PP_2], axis=2)  

    dimension_after_reshape_for_dense = num_time_steps * (d_model + d_model + d_model)
    concatenated_outputs_forM_reshaped = tf.reshape(concatenated_outputs_forM, (batch_size, dimension_after_reshape_for_dense))
    medications_hat_reshaped = self.M_hat(concatenated_outputs_forM_reshaped)
    # === medications_hat is the output of the encoder
    medications_hat =  tf.reshape(medications_hat_reshaped, (batch_size, num_time_steps, d_model))
    
    concatenated_outputs_forD_reshaped = tf.reshape(concatenated_outputs_forD, (batch_size, dimension_after_reshape_for_dense))
    diagnoses_hat_reshaped = self.D_hat(concatenated_outputs_forD_reshaped)
    # === diagnoses_hat is the output of the encoder    
    diagnoses_hat =  tf.reshape(diagnoses_hat_reshaped, (batch_size, num_time_steps, d_model)) 

    concatenated_outputs_forP_reshaped = tf.reshape(concatenated_outputs_forP, (batch_size, dimension_after_reshape_for_dense))
    procedures_hat_reshaped = self.P_hat(concatenated_outputs_forP_reshaped)
    # === diagnoses_hat is the output of the encoder    
    procedures_hat =  tf.reshape(procedures_hat_reshaped, (batch_size, num_time_steps, d_model))        
    return medications_hat, diagnoses_hat, procedures_hat

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x_1, x_2, x_3, training, mask):
    # pdb.set_trace()
    seq_len = tf.shape(x_1)[1]
    x_1 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x_2 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x_3 *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x_1 += self.pos_encoding[:, :seq_len, :]
    x_2 += self.pos_encoding[:, :seq_len, :]
    x_3 += self.pos_encoding[:, :seq_len, :]
    x_1 = self.dropout(x_1, training=training)
    x_2 = self.dropout(x_2, training=training)
    x_3 = self.dropout(x_3, training=training)
    # ==== Pluging the inputs (medications, diagnoses) to the encoder layer and repeat for all layers
    for i in range(self.num_layers):
      x_1, x_2, x_3 = self.enc_layers[i](x_1, x_2, x_3, training, mask)
    return x_1, x_2, x_3 

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    self.medications_map = tf.keras.layers.Dense(d_model)
    self.diagnoses_map = tf.keras.layers.Dense(d_model)
    self.procedures_map = tf.keras.layers.Dense(d_model)
    
    self.w5 = tf.keras.layers.Dense(num_classes, kernel_regularizer= regularizers.l2(regu_factor), activity_regularizer=regularizers.l2(regu_factor))
 
  def call(self, medications, diagnoses, procedures, demog_info, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    batch_size = tf.shape(medications)[0]
    # pdb.set_trace()
    # === an initial map for medications and diagnoses streams to match their dimensions
    medications = self.medications_map(medications)
    diagnoses = self.diagnoses_map(diagnoses)
    procedures = self.procedures_map(procedures)
    enc_output_1, enc_output_2, enc_output_3 = self.encoder(medications, diagnoses, procedures, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # ==== concatenate the final outputs of the encoder layers and pass them through a final dense layer to create the logits
    concated_outputs= tf.concat([enc_output_1, enc_output_2, enc_output_3, demog_info], axis=2)
    concated_outputs_reshaped = tf.reshape(concated_outputs, (batch_size, num_time_steps * (d_model + d_model + d_model + d_demogs)))
    logits=self.w5(concated_outputs_reshaped)
    return logits


optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def my_loss(real, pred):
    loss=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=real)    
    return tf.reduce_mean(loss)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)


def train_step(medications, diagnoses, demog_info, tar):
  # pdb.set_trace()
  enc_padding_mask = create_padding_mask(medications, diagnoses, procedures)
  with tf.GradientTape() as tape:
    logits = transformer(medications, diagnoses, procedures, demog_info, 
                                 True, 
                                 enc_padding_mask=enc_padding_mask, 
                                 look_ahead_mask=None, 
                                 dec_padding_mask=None)
    predictions = tf.nn.softmax(logits)
    myLoss= my_loss(tar, logits)
    myLoss_temp=myLoss
  gradients = tape.gradient(myLoss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  return predictions, myLoss_temp, tar, logits#, tar_real, enc_padding_mask, combined_mask, dec_padding_mask

#===================== READING VALIDATION DATA ====================================
# pdb.set_trace()

validationset_meds = ReadingData(path_t=validation_filename_meds)
validationset_diags = ReadingData(path_t=validation_filename_diags)
validationset_procs = ReadingData(path_t=validation_filename_procs)
validation_demogs = ReadingData(path_t=validation_filename_demogs)
#===== Validation meds
validation_data = validationset_meds.data
validation_data_ar = np.array(validation_data)
validation_meds_enrolids= validation_data_ar[:,0]
validation_data_ar_reshaped = np.reshape(validation_data_ar[:,one:-5],(len(validation_data_ar), num_time_steps, d_meds))   
validationset_labels = validation_data_ar[:,-5:-3]
medications_validation = tf.convert_to_tensor(validation_data_ar_reshaped, np.float32)
validation_meds_enrolids = validation_data_ar[:,0]


validation_split_k_all = find_divisables(len(validation_data_ar))
validation_split_k = int(len(validation_data_ar)/validation_split_k_all[1])

#==== Validation diags
validation_data = validationset_diags.data 
validation_data_ar=np.array(validation_data)
validation_diags_enrolids= validation_data_ar[:,0]
validation_data_ar_reshaped=np.reshape(validation_data_ar[:,one:-5],(len(validation_data_ar), num_time_steps, d_diags ))   
diagnoses_validation = tf.convert_to_tensor(validation_data_ar_reshaped, np.float32)
validation_diags_enrolids = validation_data_ar[:,0]

#==== Validation procs
validation_data = validationset_procs.data 
validation_data_ar=np.array(validation_data)
validation_diags_enrolids= validation_data_ar[:,0]
validation_data_ar_reshaped=np.reshape(validation_data_ar[:,one:-5],(len(validation_data_ar), num_time_steps, d_procs ))   
procedures_validation = tf.convert_to_tensor(validation_data_ar_reshaped, np.float32)
validation_procs_enrolids = validation_data_ar[:,0]


#==== Validation demographic informations
validation_data = validation_demogs.data
validation_data_ar=np.array(validation_data)
validation_demogs_enrolids= validation_data_ar[:,0]
batch_x_ar_reshaped=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_demogs+1))   
validation_demog_info=tf.convert_to_tensor(batch_x_ar_reshaped[:,:,one:], np.float32)



if (validation_meds_enrolids != validation_diags_enrolids).all() or (validation_meds_enrolids != validation_demogs_enrolids).all() or (validation_meds_enrolids != validation_procs_enrolids).all():
  print("Error: enrolids don't match")
  pdb.set_trace()
 



# ========= Reading training data
trainset_meds = ReadingData(path_t=train_filename_meds)
trainset_diags = ReadingData(path_t=train_filename_diags)
trainset_procs = ReadingData(path_t=train_filename_procs)
trainset_demogs = ReadingData(path_t=train_filename_demogs)


#===== Training meds
train_data = trainset_meds.data
train_data_ar = np.array(train_data)
train_meds_enrolids= train_data_ar[:,0]
train_data_ar_reshaped = np.reshape(train_data_ar[:,one:-5],(len(train_data_ar), num_time_steps, d_meds))   
trainset_labels = train_data_ar[:,-5:-3]
medications_train = tf.convert_to_tensor(train_data_ar_reshaped, np.float32)
train_meds_enrolids = train_data_ar[:,0]

#==== Training diags
train_data = trainset_diags.data 
train_data_ar=np.array(train_data)
train_diags_enrolids= train_data_ar[:,0]
train_data_ar_reshaped=np.reshape(train_data_ar[:,one:-5],(len(train_data_ar), num_time_steps, d_diags ))   
diagnoses_train = tf.convert_to_tensor(train_data_ar_reshaped, np.float32)
train_diags_enrolids = train_data_ar[:,0]

#==== Training procs
train_data = trainset_procs.data 
train_data_ar=np.array(train_data)
train_diags_enrolids= train_data_ar[:,0]
train_data_ar_reshaped=np.reshape(train_data_ar[:,one:-5],(len(train_data_ar), num_time_steps, d_procs ))   
procedures_train = tf.convert_to_tensor(train_data_ar_reshaped, np.float32)
train_procs_enrolids = train_data_ar[:,0]


#==== Training demographics
train_data = trainset_demogs.data
train_data_ar=np.array(train_data)
train_demogs_enrolids= train_data_ar[:,0]
batch_x_ar_reshaped=np.reshape(train_data_ar[:,one:],(len(train_data_ar), num_time_steps, d_demogs+1))   
train_demog_info=tf.convert_to_tensor(batch_x_ar_reshaped[:,:,one:], np.float32)


train_set_shape = np.shape(trainset_meds.data)
real_labels=np.zeros((BATCH_SIZE, num_classes), dtype=np.float32)
random.seed(time.clock())
model_number=123456#random.randint(1, 1000000)
print("TRAINING  MULTI-STREAM MODEL NUMBER: ", model_number)
print("Number of epochs is: ,", EPOCHS)
checkpoint_path = "saved_models/MUPOD/checkpoints/trained_model_" + str(model_number) 
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=EPOCHS)
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
validation_iterative_header = "Validation Accuracy, Validation Precision, Validation Recall, Validation F1-Score, Validation Specificity, Validation TP, Validation TN , Validation FP, Validation FN, validation auc\n"
with open("results/MUPOD/training_loss_twostreamT_thresholding_"+str(model_number)+".csv", 'w') as loss_file, open("results/MUPOD/validation_loss_twostreamT_thresholding"+str(model_number)+".csv", 'w') as val_loss_file:#, open("results/MUPOD/train_loss_afterepoch_twostreamT_thresholding"+str(model_number)+".csv", 'a') as trn_loss_file, open("results/MUPOD/validation_iterative_performance_thresholding_"+str(model_number)+".csv",'w') as val_perf_file:
    # val_perf_file.write("".join(["".join(x) for x in validation_iterative_header]))
    for epoch in range(EPOCHS):
        print("========== Epoch: ", epoch)
        # pdb.set_trace()
        step=1
        while step * BATCH_SIZE < train_set_shape[0]:
            # pdb.set_trace()
            # ==== medication stream
            batch_x = trainset_meds.next(BATCH_SIZE)
            batch_x_ar=np.array(batch_x)
            real_labels = batch_x_ar[:,-5:-3]
            meds_enrolids= batch_x_ar[:,0]
            batch_x_ar_reshaped=np.reshape(batch_x_ar[:,one:-5],(BATCH_SIZE, num_time_steps, d_meds))   
            medications=tf.convert_to_tensor(batch_x_ar_reshaped, np.float32)
            real_labels=tf.convert_to_tensor(real_labels, np.float32)
            
            # ==== Diagnoses stream
            batch_x = trainset_diags.next(BATCH_SIZE)
            batch_x_ar=np.array(batch_x)
            diags_enrolids= batch_x_ar[:,0]
            batch_x_ar_reshaped=np.reshape(batch_x_ar[:,one:-5],(BATCH_SIZE, num_time_steps, d_diags))   
            diagnoses=tf.convert_to_tensor(batch_x_ar_reshaped, np.float32)

            # ==== Procedures stream
            batch_x = trainset_procs.next(BATCH_SIZE)
            batch_x_ar=np.array(batch_x)
            procs_enrolids= batch_x_ar[:,0]
            batch_x_ar_reshaped=np.reshape(batch_x_ar[:,one:-5],(BATCH_SIZE, num_time_steps, d_procs))   
            procedures=tf.convert_to_tensor(batch_x_ar_reshaped, np.float32)
                        
            #==== Demographic info
            batch_x = trainset_demogs.next(BATCH_SIZE)
            batch_x_ar=np.array(batch_x)
            demogs_enrolids= batch_x_ar[:,0]
            batch_x_ar_reshaped=np.reshape(batch_x_ar[:,one:],(BATCH_SIZE, num_time_steps, d_demogs+1))   
            demog_info=tf.convert_to_tensor(batch_x_ar_reshaped[:,:,one:], np.float32)            
            
            #=== double check enrolids
            # pdb.set_trace()
            if (meds_enrolids != diags_enrolids ).all() or (meds_enrolids != procs_enrolids).all() or (meds_enrolids != demogs_enrolids).all():
              print("Error: enrolids don't match")
              pdb.set_trace()
            
            predictions_temp, loss, tar, logits_temp = train_step(medications, diagnoses, demog_info, real_labels)
            loss_file.write(str(loss.numpy()))
            loss_file.write('\n')           
            step = step + 1  
        # Measure validation loss every 5 epochs                          
        # if epoch % 5 == 0:
            # Spliting the validation set to smaller sets to save memory space
            # procedures_validation_split=tf.split(value=procedures_validation, num_or_size_splits=validation_split_k)
            # diagnoses_validation_split=tf.split(value=diagnoses_validation, num_or_size_splits=validation_split_k)
            # medications_validation_split=tf.split(value=medications_validation, num_or_size_splits=validation_split_k)
            # validation_demog_info_split=tf.split(value=validation_demog_info, num_or_size_splits=validation_split_k)
            # logits_validation_all = list()
            # for i in range(len(diagnoses_validation_split)):
            #   val_enc_mask = create_padding_mask(medications_validation_split[i], diagnoses_validation_split[i], procedures_validation_split[i])
            #   logits_validation = transformer(medications_validation_split[i], diagnoses_validation_split[i], procedures_validation_split[i], validation_demog_info_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
            #   logits_validation_all.extend(logits_validation)
            # val_loss= my_loss(validationset_labels, logits_validation_all)
            # val_loss_file.write(str(val_loss.numpy()))
            # val_loss_file.write("\n")

# Saving the trained model
ckpt_save_path = ckpt_manager.save()
# pdb.set_trace()
procedures_validation_split=tf.split(value=procedures_validation, num_or_size_splits=validation_split_k)
diagnoses_validation_split=tf.split(value=diagnoses_validation, num_or_size_splits=validation_split_k)
medications_validation_split=tf.split(value=medications_validation, num_or_size_splits=validation_split_k)
validation_demog_info_split=tf.split(value=validation_demog_info, num_or_size_splits=validation_split_k)
logits_validation_all = list()
for i in range(len(diagnoses_validation_split)):
  val_enc_mask = create_padding_mask(medications_validation_split[i], diagnoses_validation_split[i], procedures_validation_split[i])
  logits_validation = transformer(medications_validation_split[i], diagnoses_validation_split[i], procedures_validation_split[i], validation_demog_info_split[i], False, enc_padding_mask=val_enc_mask, look_ahead_mask=None, dec_padding_mask=None)
  logits_validation_all.extend(logits_validation)
predictions_validation_soft=tf.nn.softmax(logits_validation_all)
np.savetxt('results/MUPOD/validation_enrolids_multistream.csv', validation_meds_enrolids, delimiter=',')
np.savetxt('results/MUPOD/predictions_validation_soft_multistream_'+'_'+str(model_number)+'.csv', predictions_validation_soft)
np.savetxt('results/MUPOD/logits_validation_multistream_'+'_'+str(model_number)+'.csv', logits_validation)

#============ Thresholding 
probabilities_validation_pos=predictions_validation_soft.numpy()[:,0]
current_threshold_temp=0
thresholding_results=np.zeros((num_thresholds, 11)) 

# fpr, tpr, thresholds = metrics.roc_curve(validationset_labels[:,0], probabilities_validation_pos, pos_label=1)
# validation_auc = metrics.auc(fpr, tpr)
validation_auc=metrics.roc_auc_score(validationset_labels[:,0], probabilities_validation_pos)       
for thresh_index in range(num_thresholds):
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

header_results_filename= "Model Number (TWO STREAM), Learning Rate, Number of Layeres, Number of heads, Batch Size, Dropout Rate, Number of EPOCHS, Validation Accuracy, Validation Precision, Validation Recall, Validation F1-Score, Validation Specificity, Validation TP, Validation TN , Validation FP, Validation FN, Valication AUC, validation optimum threshold, regularization factor\n"
print("=================== TWO STREAM MODEL======================")
#pdb.set_trace()
with open("results/MUPOD/validation_results_twostreamT_thresholding.csv", 'a') as results_file:
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
