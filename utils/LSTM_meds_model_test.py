from __future__ import print_function

import os
import tensorflow as tf     
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import random
import numpy as np
import csv
import pdb
from sklearn import metrics
import pandas as pd
import tensorflow_addons as tfa

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def convert_1d_2d(current_batch_meds
                , current_batch_metadata
                , num_time_steps
                , d_meds
                , batch_size):
    one=1
    current_batch_meds_ar=np.array(current_batch_meds)
    meds_enrolids = current_batch_meds_ar[:,0]
    medications=np.reshape(current_batch_meds_ar[:,one:],(len(current_batch_meds_ar), num_time_steps, d_meds+1))  # +1 for visit date 
    
    
    labels_train = np.zeros((batch_size, 2))
    labels_temp = np.array(current_batch_metadata)[:,-1]                
    labels_train[:,0] = labels_temp
    labels_train[labels_temp == 0, 1] = 1
    labels_enrolids = np.array(current_batch_metadata)[:,0]


    if not(sum(meds_enrolids == labels_enrolids) == len(meds_enrolids)):
        print("Error: test enrolids don't match!")
        pdb.set_trace()
    

    current_batch_length=[]
    for i in range(len(medications)):
        current_batch_length.append(find_length(medications[i][:,one:]))

    return medications, labels_train, current_batch_length

def read_a_batch(file, batch_size):
    # pdb.set_trace()
    line_counter = 1
    eof_reached = 0
    current_line = file.readline()
    if current_line.split(',')[0] == 'ENROLID':
       current_line = file.readline() 
    # If it is EOF start from beginning
    if current_line == '':
        eof_reached = 1
        file.seek(0)
        current_line = file.readline()
        if current_line.split(',')[0] == 'ENROLID':
            current_line = file.readline() 
    current_line = current_line.split(',')
    current_line = [float(i) for i in current_line]
    current_batch = [] 
    current_batch.append(current_line)   
    while line_counter < batch_size:
        current_line = file.readline()
        if current_line.split(',')[0] == 'ENROLID':
            current_line = file.readline() 
        if current_line == '':
            eof_reached = 1
            file.seek(0)        
            current_line = file.readline() 
            if current_line.split(',')[0] == 'ENROLID':
                current_line = file.readline() 
        current_line = current_line.split(',')
        current_line = [float(i) for i in current_line]        
        current_batch.append(current_line)   
        line_counter += 1
    return current_batch, eof_reached



def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drp):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Embedding 
    # ==========
    # batch_size=tf.shape(x)[0]
    # x_embeded_1 =  tf.matmul(x,weights_emb)
    # x_sum_rows = tf.math.reduce_sum(x, axis=2)
    
    # x_sum_rows_reshaped = tf.reshape(x_sum_rows,(batch_size, seq_max_len,1))
    # x_normalized = x_embeded_1/tf.math.add(x_sum_rows_reshaped,tf.keras.backend.epsilon())
    # x = x_normalized


    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.keras.layers.LSTMCell(units=n_hidden, dropout = drp, kernel_regularizer='l2')#, activation='gelu')
    # lstm_cell = tfa.rnn.LayerNormLSTMCell(units=n_hidden, dropout = drp, kernel_regularizer='l2')
    # lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell)#,input_keep_prob=0.5, output_keep_prob=keep_prob)


    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    lstm_input = x
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs_original=outputs
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen-1)
    # Indexing
    output_before_idx =outputs
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    output_after_idx =outputs
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out'], states, outputs, outputs_original, output_before_idx, output_after_idx, lstm_input



# Parameters
def find_length(sequence):
    # pdb.set_trace()
    length=0
    for i in range(len(sequence)):
        if sum(sequence[i]) != 0:
            length=i
    return (length+1)

def find_beg_end(sequence):
    beginning=-1
    end=-1
    for i in range(len(sequence)):
        if sum(sequence[i]) != 0:
            end=i
        if sum(sequence[i]) != 0 and end ==-1:
            beginning= i    
    return beginning+1, end+1    
def main(idx, drp, representing, epochs, reg_coeff, learning_rt,n_hid, batch_sz,  trained_model_path, test_meds_filename, test_metadata_filename):
    tf.compat.v1.reset_default_graph()
    # pdb.set_trace()
    print("Creating and training the LSTM-meds model ...")
    learning_rate = learning_rt
    training_iters_low = 10000
    batch_size = batch_sz
    display_step = 10
    loss_threshold = 0.0001
    n_hidden = n_hid 
    num_classes = 2 
    num_time_steps=138
    seq_max_len = 138
    d_meds = 50
    d_embedding_meds = 16
    # embeding parameters
    one = 1
    zero = 0
    # Read test set using pandas
    # pdb.set_trace()
    # print('===============================================')
    # print('Warning: you are reading first 200 rows.')
    test_meds = pd.read_csv(test_meds_filename, skiprows=1, header=None)#, nrows=200)
    test_meta = pd.read_csv(test_metadata_filename)#, nrows=200)
    val_medications=np.reshape(test_meds.values[:,one:],(test_meds.shape[0], num_time_steps, d_meds+1))  # +1 for visit date 
    test_labels = np.zeros((val_medications.shape[0], 2))
    test_labels[:,0] = test_meta[' Label']  
    test_labels[:,1] = 1-test_labels[:,0]
    

    val_length=[]
    for i in range(len(val_medications)):
        val_length.append(find_length(val_medications[i][:,one:]))
    # pdb.set_trace()    

    # Check if enrolids match
    if not(sum(test_meds.loc[:,0] == test_meta['ENROLID'])==test_meds.shape[0]):
        pdb.set_trace()
        print('Warning: ENROLID mismatch.')

# tf Graph input
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder("float", [None, num_time_steps, d_meds])  # input sequence
    y = tf.compat.v1.placeholder("float", [None, num_classes])       # labels
# A placeholder for indicating each sequence length
    seqlen = tf.compat.v1.placeholder(tf.int32, [None])               # sequence length

# Define weights
    weights = {
        'out': tf.Variable(tf.random.normal([n_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random.normal([num_classes]))
    }

    # weights_emb = tf.Variable(tf.random.normal([d_meds, d_embedding_meds]), trainable=True, name = 'weight_for_emb')
    
    pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input = dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drp)

# Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #+  (reg_coeff * tf.reduce_sum(tf.nn.l2_loss(weights['out'])))
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pred_arg=tf.argmax(pred,1)
    y_arg=tf.argmax(y,1)
    softmax_predictions = tf.nn.softmax(pred)

# Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        print("Loading the saved LSTM-meds model under:")
        print(trained_model_path)
        # pdb.set_trace() 
        saver.restore(sess, tf.train.latest_checkpoint(trained_model_path))
        print("Model restored.")

        [y_arg_test, softmax_predictions_test, pred_arg_test] =sess.run([y_arg, softmax_predictions, pred_arg], feed_dict={x: val_medications[:,:,one:], y: test_labels,seqlen: val_length})

        test_auc=metrics.roc_auc_score(test_labels[:,0], np.array(softmax_predictions_test)[:,0])  
        
        print("=================================")
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(pred_arg_test)):
            if(pred_arg_test[i]==1 and y_arg_test[i]==1):
                tn=tn+1
            elif(pred_arg_test[i]==0 and y_arg_test[i]==0):
                tp=tp+1
            elif(pred_arg_test[i]==0 and y_arg_test[i]==1):
                fp=fp+1
            elif(pred_arg_test[i]==1 and y_arg_test[i]==0):
                fn=fn+1          
        if( (tp+fp)==0):
            precision=0
        else:
            precision=tp/(tp+fp)
        
        if (tp+fn) == 0:
            recall = 0
        else:    
            recall=tp/(tp+fn)
        if (tp+fn)==0:
            sensitivity=0
        else:    
            sensitivity=tp/(tp+fn)
        if (tn+fp)==0:
            specificity=0
        else:
            specificity=tn/(tn+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        
        if representing == True:
            #============================== Represent patients using trained LSTM        
            # pdb.set_trace()
            print("Representing the validation data...")
            validation_meds_filename = 'outputs/validation_medications_multihot.csv'
            validation_metadata_filename = 'outputs/validation_demographics_shuffled.csv'

            test_meds_filename = 'outputs/test_medications_multihot.csv'
            test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

            train_meds_filename = 'outputs/train_medications_multihot.csv'
            train_metadata_filename = 'outputs/train_demographics_shuffled.csv'
            print("=================================")        
            # Representing validation
            with open("outputs/validation_meds_represented.csv",'w') as valid_rep_file, open(validation_meds_filename) as validation_meds_file, open(validation_metadata_filename) as validation_meta_file:
                next(validation_meta_file)
                next(validation_meds_file)
                # line_counter=0
                for line_med in validation_meds_file:
                    # line_counter+=1
                    # if line_counter>100:
                    #     break
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_meta = validation_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                    
                    labels_temp = np.array(line_meta)[-1]  
                    if labels_temp == 1:
                        labels_2d_temp = [1, 0]
                        # validation_labels.append(labels_2d_temp)              
                    elif labels_temp == 0:
                        labels_2d_temp = [0, 1]
                        # validation_labels.append(labels_2d_temp)    
                    else:
                        pdb.set_trace()    
                        print('Warning')
                    
                    # pdb.set_trace()    
                    length = [find_length(medications[0,:,one:])]
                    beginning, end = find_beg_end(medications[0,:,one:])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: medications[:,:,one:], y: [labels_2d_temp],seqlen: length})            
                    array_temp=np.array(outputs_original_temp).flatten()
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, array_temp)))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, labels_2d_temp)))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(beginning)))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(end)))                    
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write('\n')

            # Representing testing
            print("Representing the test data...")            
            with open("outputs/test_meds_represented.csv",'w') as test_rep_file, open(test_meds_filename) as test_meds_file, open(test_metadata_filename) as test_meta_file:
                next(test_meta_file)
                next(test_meds_file)
                # line_counter=0
                for line_med in test_meds_file:
                    # line_counter+=1
                    # if line_counter>100:
                    #     break                    
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_meta = test_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date                     
                    
                    labels_temp = np.array(line_meta)[-1]  
                    if labels_temp == 1:
                        labels_2d_temp = [1, 0]
                        # test_labels.append(labels_2d_temp)              
                    elif labels_temp == 0:
                        labels_2d_temp = [0, 1]
                        # test_labels.append(labels_2d_temp)    
                    else:
                        pdb.set_trace()    
                        print('Warning')                    

                    length = [find_length(medications[0,:,one:])]
                    beginning, end = find_beg_end(medications[0,:,one:])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: medications[:,:,one:], y: [labels_2d_temp],seqlen: length})            
                    array_temp=np.array(outputs_original_temp).flatten()
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, array_temp)))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, labels_2d_temp)))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(beginning)))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(end)))                    
                    test_rep_file.write(',')
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write('\n')

            # Representing train
            print("Representing the train data...")            
            with open("outputs/train_meds_represented.csv",'w') as train_rep_file, open(train_meds_filename) as train_meds_file, open(train_metadata_filename) as train_meta_file:
                next(train_meta_file)
                next(train_meds_file)
                # line_counter=0
                for line_med in train_meds_file:
                    # line_counter+=1
                    # if line_counter>100:
                    #     break
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_meta = train_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                    
                    labels_temp = np.array(line_meta)[-1]  
                    if labels_temp == 1:
                        labels_2d_temp = [1, 0]
                        # train_labels.append(labels_2d_temp)              
                    elif labels_temp == 0:
                        labels_2d_temp = [0, 1]
                        # train_labels.append(labels_2d_temp)    
                    else:
                        pdb.set_trace()    
                        print('Warning')
                    
                    length = [find_length(medications[0,:,one:])]
                    beginning, end = find_beg_end(medications[0,:,one:])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: medications[:,:,one:], y: [labels_2d_temp],seqlen: length})            
                    array_temp=np.array(outputs_original_temp).flatten()
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, array_temp)))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, labels_2d_temp)))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(beginning)))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(end)))                    
                    train_rep_file.write(',')
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write('\n')     
                # pdb.set_trace()                 
    return softmax_predictions_test, accuracy, precision, recall, sensitivity, specificity, tp, tn, fp, fn, test_auc#, accuracy_temp_train, precision_train, recall_train, sensitivity_train, specificity_train,tp_train, tn_train, fp_train, fn_train

if __name__ == "__main__": main(idx, drp,representing, epochs, reg_coeff, learning_rt, n_hid, batch_sz, trained_model_path, test_meds_filename, test_metadata_filename)


