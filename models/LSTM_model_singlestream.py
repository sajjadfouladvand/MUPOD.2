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
                , current_batch_diags
                , current_batch_procs
                , current_batch_demogs
                , current_batch_metadata
                , num_time_steps
                , d_meds
                , d_diags
                , d_procs
                , d_demogs
                , batch_size):
    one=1
    current_batch_meds_ar=np.array(current_batch_meds)
    meds_enrolids = current_batch_meds_ar[:,0]
    medications=np.reshape(current_batch_meds_ar[:,one:],(len(current_batch_meds_ar), num_time_steps, d_meds+1))  # +1 for visit date 
    
    current_batch_diags_ar=np.array(current_batch_diags)
    diags_enrolids = current_batch_diags_ar[:,0]
    diagnoses=np.reshape(current_batch_diags_ar[:,one:],(len(current_batch_diags_ar), num_time_steps, d_diags+1))  # +1 for visit date 

    current_batch_procs_ar=np.array(current_batch_procs)
    procs_enrolids = current_batch_procs_ar[:,0]
    procedures = np.reshape(current_batch_procs_ar[:,one:],(len(current_batch_procs_ar), num_time_steps, d_procs+1))  # +1 for visit date 

    current_batch_demogs_ar=np.array(current_batch_demogs)
    demogs_enrolids = current_batch_demogs_ar[:,0]
    demographics = np.reshape(current_batch_demogs_ar[:,one:],(len(current_batch_demogs_ar), num_time_steps, d_demogs+1))  # +1 for visit date 

    
    labels_train = np.zeros((batch_size, 2))
    labels_temp = np.array(current_batch_metadata)[:,-1]                
    labels_train[:,0] = labels_temp
    labels_train[labels_temp == 0, 1] = 1
    labels_enrolids = np.array(current_batch_metadata)[:,0]


    if not(sum(meds_enrolids == diags_enrolids) == sum(meds_enrolids== demogs_enrolids) == sum(meds_enrolids == procs_enrolids) == sum(meds_enrolids == labels_enrolids) == len(meds_enrolids)):
        print("Error: test enrolids don't match!")
        pdb.set_trace()
    
    meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

    current_batch_length=[]
    for i in range(len(meds_diags_procs_demogs)):
        current_batch_length.append(find_length(meds_diags_procs_demogs[i][:,:-d_demogs]))

    return meds_diags_procs_demogs, labels_train, current_batch_length

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

def line_by_line_loss(meds_file_name
                        , diags_filename
                        , procs_filename
                        , demogs_filename
                        , metadata_filename
                        , num_time_steps
                        , d_meds
                        , d_diags
                        , d_procs
                        , d_demogs
                        , sess):
    
    with open(meds_file_name) as meds_file, open(diags_filename) as diags_file, open(procs_filename) as procs_file, open(demogs_filename) as demogs_file, open(metadata_filename) as metadata_file:
        pdb.set_trace()
        one = 1
        next(metadata_file)
        labels =[]
        loss=[]

        for line_med in meds_file:
            line_med = line_med.split(',')
            line_med = [float(i) for i in line_med]

            line_diag = diags_file.readline()
            line_diag = line_diag.split(',')
            line_diag = [float(i) for i in line_diag]

            line_proc = procs_file.readline()
            line_proc = line_proc.split(',')
            line_proc = [float(i) for i in line_proc]

            line_demog = demogs_file.readline()
            line_demog = line_demog.split(',')
            line_demog = [float(i) for i in line_demog]

            line_meta = metadata_file.readline()
            line_meta = line_meta.split(',')
            line_meta = [float(i) for i in line_meta]  

            if not(line_med[0] == line_diag[0] == line_proc[0] == line_demog[0] == line_meta[0]):
                pdb.set_trace()
                print('Warning: mismatch enrolids')

            line_med_ar=np.array(line_med)
            medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
            
            line_diag_ar=np.array(line_diag)
            diagnoses=np.reshape(line_diag_ar[one:],(1, num_time_steps, d_diags+1))  # +1 for visit date 

            line_proc_ar=np.array(line_proc)
            procedures = np.reshape(line_proc_ar[one:],(1, num_time_steps, d_procs+1))  # +1 for visit date 

            line_demog_ar=np.array(line_demog)
            demographics = np.reshape(line_demog_ar[one:],(1, num_time_steps, d_demogs+1))  # +1 for visit date 

            
            
            labels_temp = np.array(line_meta)[-1]  
            if labels_temp == 1:
                labels_2d_temp = [1, 0]
                labels.append(labels_2d_temp)              
            elif labels_temp == 0:
                labels_2d_temp = [0, 1]
                labels.append(labels_2d_temp)    
            else:
                pdb.set_trace()    
                print('Warning')
                pdb.set_trace()
            
            meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

            length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
            
            loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: labels_2d_temp, seqlen: length})

            pdb.set_trace()


            print('test')

    print('test')


def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drp):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
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
def main(idx, drp, representing, epochs, reg_coeff, learning_rt,n_hid, batch_sz, train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, test_meds_filename, test_diags_filename, test_procs_filename, test_demogs_filename, test_metadata_filename):
    tf.compat.v1.reset_default_graph()
    # pdb.set_trace()
    print("Creating and training the single stream LSTM model ...")
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
    d_diags = 60
    d_procs = 83
    d_demogs = 2

    one = 1
    zero = 0
    accuracies=[]
    # Read validation set using pandas
    # pdb.set_trace()
    # print('===============================================')
    # print('Warning: you are reading first 200 rows.')
    test_meds = pd.read_csv(test_meds_filename, skiprows=1, header=None)#, nrows=200)
    test_diags = pd.read_csv(test_diags_filename, skiprows=1, header=None)#, nrows=200)
    test_procs = pd.read_csv(test_procs_filename, skiprows=1, header=None)#, nrows=200)
    test_demogs = pd.read_csv(test_demogs_filename, header=None)#, nrows=200)
    test_meta = pd.read_csv(test_metadata_filename)#, nrows=200)


    test_medications=np.reshape(test_meds.values[:,one:],(test_meds.shape[0], num_time_steps, d_meds+1))  # +1 for visit date 
    test_diagnoses=np.reshape(test_diags.values[:,one:],(test_diags.shape[0], num_time_steps, d_diags+1))  # +1 for visit date 
    test_procedures = np.reshape(test_procs.values[:,one:],(test_procs.shape[0], num_time_steps, d_procs+1))  # +1 for visit date 
    test_demographics = np.reshape(test_demogs.values[:,one:],(test_demogs.shape[0], num_time_steps, d_demogs+1))  # +1 for visit date 

    test_labels = np.zeros((test_medications.shape[0], 2))
    test_labels[:,0] = test_meta[' Label']  
    test_labels[:,1] = 1-test_labels[:,0]
    
    test_meds_diags_procs_demogs=np.concatenate((test_medications[:,:,one:], test_diagnoses[:,:, one:], test_procedures[:,:,one:], test_demographics[:,:,one:]), axis=2)

    test_length=[]
    for i in range(len(test_meds_diags_procs_demogs)):
        test_length.append(find_length(test_meds_diags_procs_demogs[i][:,:-d_demogs]))
    # pdb.set_trace()    

    # Check if enrolids match
    if not(sum(test_meds.loc[:,0] == test_diags.loc[:,0]) == sum(test_meds.loc[:,0] == test_procs.loc[:,0]) == sum(test_meds.loc[:,0] == test_meta['ENROLID'])==test_meds.shape[0]):
        pdb.set_trace()
        print('Warning: ENROLID mismatch.')

# tf Graph input
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder("float", [None, num_time_steps, d_meds + d_diags + d_procs + d_demogs ])  # input sequence
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
        num_passed_epochs = 0           
        #with open(train_meds_filename) as train_meds_file, open(train_diags_filename) as train_diags_file, open(train_procs_filename) as train_procs_file, open(train_demogs_filename) as train_demogs_file, open(valid_meds_filename) as valid_meds_file, open(valid_diags_filename) as valid_diags_file, open(valid_procs_filename) as valid_procs_file, open(valid_demogs_filename) as valid_demogs_file, open ('results/LSTM_single_stream/Loss_train_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file , open('results/LSTM_single_stream/Loss_test_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file_val:
        train_meds_file = open(train_meds_filename)
        train_diags_file = open(train_diags_filename)
        train_procs_file = open(train_procs_filename)
        train_demogs_file = open(train_demogs_filename)
        train_metadata_file = open(train_metadata_filename)
        
        train_loss_file = open('results/LSTM_single_stream/Loss_train_lstm_single_stream_'+str(idx)+'.csv', 'w')        
        test_loss_file = open('results/LSTM_single_stream/Loss_test_lstm_single_stream_'+str(idx)+'.csv', 'w')
        
        step=0
        ########### Overfit the model
        # pdb.set_trace()
        # for i in range(10):
        #     current_batch_meds, eof_reached_meds = read_a_batch(train_meds_file, batch_size)
        #     current_batch_diags, eof_reached_diags = read_a_batch(train_diags_file, batch_size)
        #     current_batch_procs, eof_reached_procs = read_a_batch(train_procs_file, batch_size)
        #     current_batch_demogs, eof_reached_demogs = read_a_batch(train_demogs_file, batch_size)
        #     current_batch_metadata, eof_reached_demogs = read_a_batch(train_metadata_file, batch_size)
        # pdb.set_trace()
        # epochs=100
        ###########
        while num_passed_epochs < epochs:
            # num_passed_epochs+=1 ###########
            # pdb.set_trace()
            current_batch_meds, eof_reached_meds = read_a_batch(train_meds_file, batch_size)
            current_batch_diags, eof_reached_diags = read_a_batch(train_diags_file, batch_size)
            current_batch_procs, eof_reached_procs = read_a_batch(train_procs_file, batch_size)
            current_batch_demogs, eof_reached_demogs = read_a_batch(train_demogs_file, batch_size)
            current_batch_metadata, eof_reached_demogs = read_a_batch(train_metadata_file, batch_size)           
            if not(eof_reached_meds == eof_reached_diags == eof_reached_procs == eof_reached_demogs):
                pdb.set_trace()
                print('Warning: file sizes not equal')
            if  eof_reached_meds == 1:
                # pdb.set_trace()
                num_passed_epochs += 1
                print(num_passed_epochs)

            meds_diags_procs_demogs, labels_train, current_batch_length = convert_1d_2d(current_batch_meds
                                                                                        , current_batch_diags
                                                                                        , current_batch_procs
                                                                                        , current_batch_demogs
                                                                                        , current_batch_metadata
                                                                                        , num_time_steps
                                                                                        , d_meds
                                                                                        , d_diags
                                                                                        , d_procs
                                                                                        , d_demogs
                                                                                        , batch_size)            

            sess.run(optimizer, feed_dict={x: meds_diags_procs_demogs, y: labels_train, seqlen: current_batch_length})  
            
            # [pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input] = sess.run([pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input], feed_dict={x: meds_diags_procs_demogs, y: labels_train, seqlen: current_batch_length})  

            loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: labels_train, seqlen: current_batch_length})
            train_loss_file.write(str(loss))
            train_loss_file.write("\n")
            step+=1
            # print(step)
            # if step>2:
            #     break
            if step%100==0:
                print('The step is {} and the epoch number is {}'.format(step, num_passed_epochs))                
            # pdb.set_trace()
            # test loss calculation
            test_loss = sess.run(cost, feed_dict={x: test_meds_diags_procs_demogs, y: test_labels, seqlen: test_length})
            test_loss_file.write(str(np.mean(test_loss)))
            test_loss_file.write('\n')


        saver.save(sess, save_path='saved_models/lstm_single_stream/LSTM_meds_diags_procs_demogs_model_'+str(int(idx))+'.ckpt')
        print("Optimization Finished!")
        train_meds_file.close()
        train_diags_file.close()
        train_procs_file.close()
        train_demogs_file.close()
        train_metadata_file.close()
        train_loss_file.close()
        test_loss_file.close()
        # pdb.set_trace()
        #=================================================================
        # pdb.set_trace()
        # ==== Validating
        [y_arg_test, softmax_predictions_test, pred_arg_test] =sess.run([y_arg, softmax_predictions, pred_arg], feed_dict={x: test_meds_diags_procs_demogs, y: test_labels,seqlen: test_length})

        # with open(test_meds_filename) as test_meds_file, open(test_diags_filename) as test_diags_file, open(test_procs_filename) as test_procs_file, open(test_demogs_filename) as test_demogs_file, open(test_metadata_filename) as test_meta_file:
        #     next(test_meta_file)
        #     next(test_meds_file)
        #     next(test_diags_file)
        #     next(test_procs_file)

        #     test_labels = []
        #     test_loss = []
        #     y_arg_test = []
        #     softmax_predictions_test = []
        #     pred_arg_test = []

        #     for line_med in test_meds_file:
        #         line_med = line_med.split(',')
        #         line_med = [float(i) for i in line_med]

        #         line_diag = test_diags_file.readline()
        #         line_diag = line_diag.split(',')
        #         line_diag = [float(i) for i in line_diag]

        #         line_proc = test_procs_file.readline()
        #         line_proc = line_proc.split(',')
        #         line_proc = [float(i) for i in line_proc]

        #         line_demog = test_demogs_file.readline()
        #         line_demog = line_demog.split(',')
        #         line_demog = [float(i) for i in line_demog]

        #         line_meta = test_meta_file.readline()
        #         line_meta = line_meta.split(',')
        #         line_meta = [float(i) for i in line_meta]  

        #         if not(line_med[0] == line_diag[0] == line_proc[0] == line_demog[0] == line_meta[0]):
        #             pdb.set_trace()
        #             print('Warning: mismatch enrolids')

        #         line_med_ar=np.array(line_med)
        #         medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                
        #         line_diag_ar=np.array(line_diag)
        #         diagnoses=np.reshape(line_diag_ar[one:],(1, num_time_steps, d_diags+1))  # +1 for visit date 

        #         line_proc_ar=np.array(line_proc)
        #         procedures = np.reshape(line_proc_ar[one:],(1, num_time_steps, d_procs+1))  # +1 for visit date 

        #         line_demog_ar=np.array(line_demog)
        #         demographics = np.reshape(line_demog_ar[one:],(1, num_time_steps, d_demogs+1))  # +1 for visit date 

                
                
        #         labels_temp = np.array(line_meta)[-1]  
        #         if labels_temp == 1:
        #             labels_2d_temp = [1, 0]
        #             test_labels.append(labels_2d_temp)              
        #         elif labels_temp == 0:
        #             labels_2d_temp = [0, 1]
        #             test_labels.append(labels_2d_temp)    
        #         else:
        #             pdb.set_trace()    
        #             print('Warning')
        #             pdb.set_trace()
                
        #         meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

        #         length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
        #         # pdb.set_trace()
        #         loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp], seqlen: length})
        #         test_loss.append(loss)
        #         [y_arg_temp, softmax_predictions_temp, pred_arg_temp] =sess.run([y_arg, softmax_predictions, pred_arg], feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp],seqlen: length})
        #         y_arg_test.append(y_arg_temp[0])
        #         softmax_predictions_test.append(softmax_predictions_temp.tolist()[0])
        #         pred_arg_test.append(pred_arg_temp[0])
                
        # pdb.set_trace()
        y_arg_temp_forAUC = np.abs(np.array(y_arg_test)-1)
        # fpr, tpr, thresholds = metrics.roc_curve(y_true=y_arg_temp_forAUC, y_score=np.array(softmax_predictions_test)[:,0], pos_label=1) #== It's arg and so 1 means negative and 0 means positive
        # test_auc=metrics.auc(fpr, tpr)    
        test_auc=metrics.roc_auc_score(y_arg_temp_forAUC, np.array(softmax_predictions_test)[:,0])       
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
            validation_diags_filename = 'outputs/validation_diagnoses_multihot.csv'
            validation_procs_filename = 'outputs/validation_procedures_multihot.csv'
            validation_demogs_filename = 'outputs/validation_demographics_multihot.csv'
            validation_metadata_filename = 'outputs/validation_demographics_shuffled.csv'

            test_meds_filename = 'outputs/test_medications_multihot.csv'
            test_diags_filename = 'outputs/test_diagnoses_multihot.csv'
            test_procs_filename = 'outputs/test_procedures_multihot.csv'
            test_demogs_filename = 'outputs/test_demographics_multihot.csv'
            test_metadata_filename = 'outputs/test_demographics_shuffled.csv'

            train_meds_filename = 'outputs/train_medications_multihot.csv'
            train_diags_filename = 'outputs/train_diagnoses_multihot.csv'
            train_procs_filename = 'outputs/train_procedures_multihot.csv'
            train_demogs_filename = 'outputs/train_demographics_multihot.csv'
            train_metadata_filename = 'outputs/train_demographics_shuffled.csv'
            print("=================================")        
            # Representing validation
            with open("outputs/validation_meds_diags_procs_demogs_represented.csv",'w') as valid_rep_file, open(validation_meds_filename) as validation_meds_file, open(validation_diags_filename) as validation_diags_file, open(validation_procs_filename) as validation_procs_file, open(validation_demogs_filename) as validation_demogs_file, open(validation_metadata_filename) as validation_meta_file:
                next(validation_meta_file)
                next(validation_meds_file)
                next(validation_diags_file)
                next(validation_procs_file)

                for line_med in validation_meds_file:
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_diag = validation_diags_file.readline()
                    line_diag = line_diag.split(',')
                    line_diag = [float(i) for i in line_diag]

                    line_proc = validation_procs_file.readline()
                    line_proc = line_proc.split(',')
                    line_proc = [float(i) for i in line_proc]

                    line_demog = validation_demogs_file.readline()
                    line_demog = line_demog.split(',')
                    line_demog = [float(i) for i in line_demog]

                    line_meta = validation_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_diag[0] == line_proc[0] == line_demog[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                    
                    line_diag_ar=np.array(line_diag)
                    diagnoses=np.reshape(line_diag_ar[one:],(1, num_time_steps, d_diags+1))  # +1 for visit date 

                    line_proc_ar=np.array(line_proc)
                    procedures = np.reshape(line_proc_ar[one:],(1, num_time_steps, d_procs+1))  # +1 for visit date 

                    line_demog_ar=np.array(line_demog)
                    demographics = np.reshape(line_demog_ar[one:],(1, num_time_steps, d_demogs+1))  # +1 for visit date 

                    
                    
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
                        pdb.set_trace()
                    
                    meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

                    length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
                    beginning, end = find_beg_end(meds_diags_procs_demogs[0][:,:-d_demogs])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp],seqlen: length})            
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
            with open("outputs/test_meds_diags_procs_demogs_represented.csv",'w') as test_rep_file, open(test_meds_filename) as test_meds_file, open(test_diags_filename) as test_diags_file, open(test_procs_filename) as test_procs_file, open(test_demogs_filename) as test_demogs_file, open(test_metadata_filename) as test_meta_file:
                next(test_meta_file)
                next(test_meds_file)
                next(test_diags_file)
                next(test_procs_file)

                for line_med in test_meds_file:
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_diag = test_diags_file.readline()
                    line_diag = line_diag.split(',')
                    line_diag = [float(i) for i in line_diag]

                    line_proc = test_procs_file.readline()
                    line_proc = line_proc.split(',')
                    line_proc = [float(i) for i in line_proc]

                    line_demog = test_demogs_file.readline()
                    line_demog = line_demog.split(',')
                    line_demog = [float(i) for i in line_demog]

                    line_meta = test_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_diag[0] == line_proc[0] == line_demog[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                    
                    line_diag_ar=np.array(line_diag)
                    diagnoses=np.reshape(line_diag_ar[one:],(1, num_time_steps, d_diags+1))  # +1 for visit date 

                    line_proc_ar=np.array(line_proc)
                    procedures = np.reshape(line_proc_ar[one:],(1, num_time_steps, d_procs+1))  # +1 for visit date 

                    line_demog_ar=np.array(line_demog)
                    demographics = np.reshape(line_demog_ar[one:],(1, num_time_steps, d_demogs+1))  # +1 for visit date 

                    
                    
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
                        pdb.set_trace()
                    
                    meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

                    length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
                    beginning, end = find_beg_end(meds_diags_procs_demogs[0][:,:-d_demogs])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp],seqlen: length})            
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
            with open("outputs/train_meds_diags_procs_demogs_represented.csv",'w') as train_rep_file, open(train_meds_filename) as train_meds_file, open(train_diags_filename) as train_diags_file, open(train_procs_filename) as train_procs_file, open(train_demogs_filename) as train_demogs_file, open(train_metadata_filename) as train_meta_file:
                next(train_meta_file)
                next(train_meds_file)
                next(train_diags_file)
                next(train_procs_file)

                for line_med in train_meds_file:
                    line_med = line_med.split(',')
                    line_med = [float(i) for i in line_med]

                    line_diag = train_diags_file.readline()
                    line_diag = line_diag.split(',')
                    line_diag = [float(i) for i in line_diag]

                    line_proc = train_procs_file.readline()
                    line_proc = line_proc.split(',')
                    line_proc = [float(i) for i in line_proc]

                    line_demog = train_demogs_file.readline()
                    line_demog = line_demog.split(',')
                    line_demog = [float(i) for i in line_demog]

                    line_meta = train_meta_file.readline()
                    line_meta = line_meta.split(',')
                    line_meta = [float(i) for i in line_meta]  

                    if not(line_med[0] == line_diag[0] == line_proc[0] == line_demog[0] == line_meta[0]):
                        pdb.set_trace()
                        print('Warning: mismatch enrolids')
                    current_patient_enrolid = line_med[0]    
                    line_med_ar=np.array(line_med)
                    medications=np.reshape(line_med_ar[one:],(1, num_time_steps, d_meds+1))  # +1 for visit date 
                    
                    line_diag_ar=np.array(line_diag)
                    diagnoses=np.reshape(line_diag_ar[one:],(1, num_time_steps, d_diags+1))  # +1 for visit date 

                    line_proc_ar=np.array(line_proc)
                    procedures = np.reshape(line_proc_ar[one:],(1, num_time_steps, d_procs+1))  # +1 for visit date 

                    line_demog_ar=np.array(line_demog)
                    demographics = np.reshape(line_demog_ar[one:],(1, num_time_steps, d_demogs+1))  # +1 for visit date 

                    
                    
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
                        pdb.set_trace()
                    
                    meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

                    length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
                    beginning, end = find_beg_end(meds_diags_procs_demogs[0][:,:-d_demogs])
                    # pdb.set_trace()
                    [states_temp, outputs_original_temp] =sess.run([states, outputs_original], feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp],seqlen: length})            
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

if __name__ == "__main__": main(idx, drp,representing, epochs, reg_coeff, learning_rt, n_hid, batch_sz, train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, test_meds_filename, test_diags_filename, test_procs_filename, test_demogs_filename, test_metadata_filename)


