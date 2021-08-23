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

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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
        print("Error: validation enrolids don't match!")
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


def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell)#,input_keep_prob=0.5, output_keep_prob=keep_prob)


    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    lstm_input = x
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
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
def main(idx, representing, epochs, reg_coeff, learning_rt,n_hid, batch_sz, train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename):
    tf.reset_default_graph() 
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
    d_meds = 94
    d_diags = 284
    d_procs = 243
    d_demogs = 2
    one = 1
    zero = 0
    accuracies=[]

# tf Graph input
    x = tf.placeholder("float", [None, num_time_steps, d_meds + d_diags + d_procs + d_demogs ])  # input sequence
    y = tf.placeholder("float", [None, num_classes])       # labels
# A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])               # sequence length

# Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }


    pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input = dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden)

# Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) #+  (reg_coeff * tf.reduce_sum(tf.nn.l2_loss(weights['out'])))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pred_arg=tf.argmax(pred,1)
    y_arg=tf.argmax(y,1)
    softmax_predictions = tf.nn.softmax(pred)

# Initializing the variables
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        num_passed_epochs = 0           
        #with open(train_meds_filename) as train_meds_file, open(train_diags_filename) as train_diags_file, open(train_procs_filename) as train_procs_file, open(train_demogs_filename) as train_demogs_file, open(valid_meds_filename) as valid_meds_file, open(valid_diags_filename) as valid_diags_file, open(valid_procs_filename) as valid_procs_file, open(valid_demogs_filename) as valid_demogs_file, open ('results/LSTM_single_stream/Loss_train_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file , open('results/LSTM_single_stream/Loss_validation_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file_val:
        train_meds_file = open(train_meds_filename)
        train_diags_file = open(train_diags_filename)
        train_procs_file = open(train_procs_filename)
        train_demogs_file = open(train_demogs_filename)
        train_metadata_file = open(train_metadata_filename)
        train_loss_file = open('results/LSTM_single_stream/Loss_train_lstm_single_stream_'+str(idx)+'.csv', 'w')
        

        validation_meds_file = open(validation_meds_filename)
        validation_diags_file = open(validation_diags_filename)
        validation_procs_file = open(validation_procs_filename)
        validation_demogs_file = open(validation_demogs_filename)
        validation_metadata_file = open(validation_metadata_filename)
        validation_loss_file = open('results/LSTM_single_stream/Loss_validation_lstm_single_stream_'+str(idx)+'.csv', 'w')
        step=0
        while num_passed_epochs < epochs:
            current_batch_meds, eof_reached_meds = read_a_batch(train_meds_file, batch_size)
            current_batch_diags, eof_reached_diags = read_a_batch(train_diags_file, batch_size)
            current_batch_procs, eof_reached_procs = read_a_batch(train_procs_file, batch_size)
            current_batch_demogs, eof_reached_demogs = read_a_batch(train_demogs_file, batch_size)
            current_batch_metadata, eof_reached_demogs = read_a_batch(train_metadata_file, batch_size)
            
            if not(eof_reached_meds == eof_reached_diags == eof_reached_procs == eof_reached_demogs):
                pdb.set_trace()
                print('Warning: file sizes not equal')
            if  eof_reached_meds == 1:
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
            loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: labels_train, seqlen: current_batch_length})
            train_loss_file.write(str(loss))
            train_loss_file.write("\n")
            step+=1
            # print(step)
            if step % 100 == 0:
                print('the step is {}'.format(step))
                print('The epoch number is {}'.format(num_passed_epochs))
            # pdb.set_trace()
            # ====validation loss calculation
            # with open(validation_meds_filename) as val_meds_file, open(validation_diags_filename) as val_diags_file, open(validation_procs_filename) as val_procs_file, open(validation_demogs_filename) as val_demogs_file, open(validation_metadata_filename) as val_meta_file:
            #     next(val_meta_file)
            #     val_labels = []
            #     val_loss = []
            #     for line_med in val_meds_file:
            #         line_med = line_med.split(',')
            #         line_med = [float(i) for i in line_med]

            #         line_diag = val_diags_file.readline()
            #         line_diag = line_diag.split(',')
            #         line_diag = [float(i) for i in line_diag]

            #         line_proc = val_procs_file.readline()
            #         line_proc = line_proc.split(',')
            #         line_proc = [float(i) for i in line_proc]

            #         line_demog = val_demogs_file.readline()
            #         line_demog = line_demog.split(',')
            #         line_demog = [float(i) for i in line_demog]

            #         line_meta = val_meta_file.readline()
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
            #             val_labels.append(labels_2d_temp)              
            #         elif labels_temp == 0:
            #             labels_2d_temp = [0, 1]
            #             val_labels.append(labels_2d_temp)    
            #         else:
            #             pdb.set_trace()    
            #             print('Warning')
            #             pdb.set_trace()
                    
            #         meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

            #         length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
            #         # pdb.set_trace()
            #         loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp], seqlen: length})
            #         val_loss.append(loss)
            # validation_loss_file.write(str(np.mean(val_loss)))
            # validation_loss_file.write('\n')
            # print('end')
        # pdb.set_trace()
        saver.save(sess, save_path='saved_models/lstm_single_stream/LSTM_meds_diags_procs_demogs_model_'+str(int(idx))+'.ckpt')
        print("Optimization Finished!")
        #=================================================================
        # pdb.set_trace()
        # ==== Validating
        with open(validation_meds_filename) as validation_meds_file, open(validation_diags_filename) as validation_diags_file, open(validation_procs_filename) as validation_procs_file, open(validation_demogs_filename) as validation_demogs_file, open(validation_metadata_filename) as validation_meta_file:
            next(validation_meta_file)
            validation_labels = []
            validation_loss = []
            y_arg_validation = []
            softmax_predictions_validation = []
            pred_arg_validation = []

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
                    validation_labels.append(labels_2d_temp)              
                elif labels_temp == 0:
                    labels_2d_temp = [0, 1]
                    validation_labels.append(labels_2d_temp)    
                else:
                    pdb.set_trace()    
                    print('Warning')
                    pdb.set_trace()
                
                meds_diags_procs_demogs=np.concatenate((medications[:,:,one:], diagnoses[:,:, one:], procedures[:,:,one:], demographics[:,:,one:]), axis=2)

                length = [find_length(meds_diags_procs_demogs[0,:,:-d_demogs])]
                # pdb.set_trace()
                loss = sess.run(cost, feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp], seqlen: length})
                validation_loss.append(loss)
                [y_arg_temp, softmax_predictions_temp, pred_arg_temp] =sess.run([y_arg, softmax_predictions, pred_arg], feed_dict={x: meds_diags_procs_demogs, y: [labels_2d_temp],seqlen: length})
                y_arg_validation.append(y_arg_temp[0])
                softmax_predictions_validation.append(softmax_predictions_temp.tolist()[0])
                pred_arg_validation.append(pred_arg_temp[0])
                
        # pdb.set_trace()
        y_arg_temp_forAUC = np.abs(np.array(y_arg_validation)-1)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_arg_temp_forAUC, y_score=np.array(softmax_predictions_validation)[:,0], pos_label=1) #== It's arg and so 1 means negative and 0 means positive
        validation_auc=metrics.auc(fpr, tpr)         
        print("=================================")
        tp=0
        tn=0
        fp=0
        fn=0
        for i in range(len(pred_arg_validation)):
            if(pred_arg_validation[i]==1 and y_arg_validation[i]==1):
                tn=tn+1
            elif(pred_arg_validation[i]==0 and y_arg_validation[i]==0):
                tp=tp+1
            elif(pred_arg_validation[i]==0 and y_arg_validation[i]==1):
                fp=fp+1
            elif(pred_arg_validation[i]==1 and y_arg_validation[i]==0):
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
            pdb.set_trace()
            print("Representing the data...")
            print("=================================")        
            with open("training_meds_diags_demogs_represented.csv", 'w') as train_rep_file, open("validation_meds_diags_demogs_represented.csv",'w') as valid_rep_file, open("testing_meds_diags_demogs_represented.csv", 'w') as test_rep_file:
                # if train_test == 'test':            
                train_meds_filename = 'outputs/train_medications_multihot.csv'
                train_diags_filename = 'outputs/train_diagnoses_multihot.csv'
                train_procs_filename = 'outputs/train_procedures_multihot.csv'
                train_demogs_filename = 'outputs/train_demographics_multihot.csv'
                train_metadata_filename = 'outputs/train_demographics_shuffled.csv'

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
                
                #======================== Reading validation data
                validationset_meds = ReadingData(path_t=validation_meds_filename)#, path_l=validation_demogs_filename)
                validationset_diags = ReadingData(path_t=validation_diags_filename)#, path_l=validation_demogs_filename)
                validationset_procs = ReadingData(path_t=validation_procs_filename)#, path_l=validation_demogs_filename)
                validationset_demogs = ReadingData(path_t=validation_demogs_filename)
                validation_metadata = ReadingData(path_t=validation_metadata_filename)
                
                validation_data = validationset_meds.data
                validation_data_ar=np.array(validation_data)
                val_med_enrolids = validation_data_ar[:,0]
                validation_medications=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_meds+1))  # +1 for visit date 
                
                validation_data = validationset_diags.data
                validation_data_ar=np.array(validation_data)
                val_diags_enrolids = validation_data_ar[:,0]
                validation_diagnoses=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_diags+1))  # +1 for visit date 

                validation_data = validationset_procs.data
                validation_data_ar=np.array(validation_data)
                val_procs_enrolids = validation_data_ar[:,0]
                validation_procedures=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_procs+1))  # +1 for visit date 
             

                validation_data = validationset_demogs.data
                validation_data_ar=np.array(validation_data)
                val_demogs_enrolids = validation_data_ar[:,0]
                validation_demographics=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_demogs+1))  # +1 for visit date 
                
                labels_validation = np.zeros((len(validation_metadata.data), 2))
                labels_temp = np.array(validation_metadata.data)[:,-1]
                
                labels_validation[:,0] = labels_temp
                labels_validation[labels_temp == 0, 1] = 1
                if not(sum(val_med_enrolids == val_diags_enrolids) == sum(val_med_enrolids== val_procs_enrolids) == sum(val_med_enrolids==val_demogs_enrolids) == len(val_med_enrolids)):
                    print("Error: validation enrolids don't match!")
                    pdb.set_trace()

                validation_meds_diags_procs_demogs=np.concatenate((validation_medications[:,:,one:], validation_diagnoses[:,:, one:], validation_procedures[:,:,one:], validation_demographics[:,:,one:]), axis=2)
                #========== Findinmg the sequence lengths
                #pdb.set_trace()
                # validation_seq_length=[]
                # for i in range(len(validation_meds_diags_procs_demogs)):
                #     validation_seq_length.append(find_length(validation_meds_diags_procs_demogs[i][:,:-d_demogs]))
                validation_seq_beg=[]
                validation_seq_end=[]
                for i in range(len(validation_meds_diags_procs_demogs)):
                    beginning, end = find_beg_end(validation_meds_diags_procs_demogs[i][:,:-d_demogs])
                    validation_seq_beg.append(beginning)
                    validation_seq_end.append(end)

                # validationset_meds = ReadingData(path_t=validation_meds_filename, path_l=validation_labels_filename)#,path_s=validation_lengths_filename)
                # validationset_diags = ReadingData(path_t=validation_diags_filename, path_l=validation_labels_filename)#,path_s=validation_lengths_filename)
                # validationset_demogs = ReadingData(path_t=validation_demogs_filename, path_l=validation_labels_filename)#,path_s=validation_lengths_filename)
                
                # validation_data = validationset_meds.data
                # validation_label = validationset_meds.labels
                # validation_data_ar=np.array(validation_data)
                # validation_medications=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_meds+1))   
                # validationset_labels=np.array(validation_label)

                # validation_data = validationset_diags.data
                # validation_data_ar=np.array(validation_data)
                # validation_diagnoses=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_diags+1))   

                # validation_data = validationset_demogs.data
                # validation_data_ar=np.array(validation_data)
                # validation_demographics=np.reshape(validation_data_ar[:,one:],(len(validation_data_ar), num_time_steps, d_demogs+1))   
                                
                # validation_meds_diags_demogs=np.concatenate((validation_medications[:,:,one:], validation_diagnoses[:,:, one:], validation_demographics[:,:, one:]), axis=2)

                # validation_seq_beg=[]
                # validation_seq_end=[]
                # for i in range(len(validation_medications)):
                #     beginning, end = find_beg_end(validation_meds_diags_demogs[i][:,:-d_demogs])
                #     validation_seq_beg.append(beginning)
                #     validation_seq_end.append(end)
                # # pdb.set_trace()
                # labels_validation = validationset_labels[:,one:].astype(np.float32)
                #======================== Reading training data 
                trainset_meds = ReadingData(path_t=train_meds_filename)#, path_l=train_demogs_filename)
                trainset_diags = ReadingData(path_t=train_diags_filename)#, path_l=train_demogs_filename)
                trainset_procs = ReadingData(path_t=train_procs_filename)#, path_l=train_demogs_filename)
                trainset_demogs = ReadingData(path_t=train_demogs_filename)
                train_metadata = ReadingData(path_t=train_metadata_filename)
                
                train_data = trainset_meds.data
                train_data_ar=np.array(train_data)
                val_med_enrolids = train_data_ar[:,0]
                train_medications=np.reshape(train_data_ar[:,one:],(len(train_data_ar), num_time_steps, d_meds+1))  # +1 for visit date 
                
                train_data = trainset_diags.data
                train_data_ar=np.array(train_data)
                val_diags_enrolids = train_data_ar[:,0]
                train_diagnoses=np.reshape(train_data_ar[:,one:],(len(train_data_ar), num_time_steps, d_diags+1))  # +1 for visit date 

                train_data = trainset_procs.data
                train_data_ar=np.array(train_data)
                val_procs_enrolids = train_data_ar[:,0]
                train_procedures=np.reshape(train_data_ar[:,one:],(len(train_data_ar), num_time_steps, d_procs+1))  # +1 for visit date 
             

                train_data = trainset_demogs.data
                train_data_ar=np.array(train_data)
                val_demogs_enrolids = train_data_ar[:,0]
                train_demographics=np.reshape(train_data_ar[:,one:],(len(train_data_ar), num_time_steps, d_demogs+1))  # +1 for visit date 
                
                labels_train = np.zeros((len(train_metadata.data), 2))
                labels_temp = np.array(train_metadata.data)[:,-1]
                
                labels_train[:,0] = labels_temp
                labels_train[labels_temp == 0, 1] = 1
                if not(sum(val_med_enrolids == val_diags_enrolids) == sum(val_med_enrolids== val_procs_enrolids) == sum(val_med_enrolids==val_demogs_enrolids) == len(val_med_enrolids)):
                    print("Error: train enrolids don't match!")
                    pdb.set_trace()

                train_meds_diags_procs_demogs=np.concatenate((train_medications[:,:,one:], train_diagnoses[:,:, one:], train_procedures[:,:,one:], train_demographics[:,:,one:]), axis=2)
                #========== Findinmg the sequence lengths
                #pdb.set_trace()
                # train_seq_length=[]
                # for i in range(len(train_meds_diags_procs_demogs)):
                #     train_seq_length.append(find_length(train_meds_diags_procs_demogs[i][:,:-d_demogs]))
                train_seq_beg=[]
                train_seq_end=[]
                for i in range(len(train_meds_diags_procs_demogs)):
                    beginning, end = find_beg_end(train_meds_diags_procs_demogs[i][:,:-d_demogs])
                    train_seq_beg.append(beginning)
                    train_seq_end.append(end)                

                # training_meds = ReadingData(path_t=train_meds_filename, path_l=train_labels_filename)#,path_s=training_lengths_filename)
                # training_diags = ReadingData(path_t=train_diags_filename, path_l=train_labels_filename)#,path_s=training_lengths_filename)
                # training_demogs = ReadingData(path_t=train_demogs_filename, path_l=train_labels_filename)#,path_s=training_lengths_filename)

                # training_data = training_meds.data
                # training_label = training_meds.labels
                # training_data_ar=np.array(training_data)
                # training_medications=np.reshape(training_data_ar[:,one:],(len(training_data_ar), num_time_steps, d_meds+1))   
                # training_labels=np.array(training_label)

                # training_data = training_diags.data
                # training_data_ar=np.array(training_data)
                # training_diagnoses=np.reshape(training_data_ar[:,one:],(len(training_data_ar), num_time_steps, d_diags+1))   
                
                # training_data = training_demogs.data
                # training_data_ar=np.array(training_data)
                # training_demographics=np.reshape(training_data_ar[:,one:],(len(training_data_ar), num_time_steps, d_demogs+1))   

                # training_meds_diags_demogs=np.concatenate((training_medications[:,:,one:], training_diagnoses[:,:, one:], training_demographics[:,:, one:]), axis=2)
                
                # training_seq_beg=[]
                # training_seq_end=[]
                # for i in range(len(training_medications)):
                #     beginning, end = find_beg_end(training_meds_diags_demogs[i][:,:-d_demogs])
                #     training_seq_beg.append(beginning)
                #     training_seq_end.append(end)
                # labels_training = training_labels[:,one:].astype(np.float32)    
                # # pdb.set_trace()
                # #======================== Reading testing data 
                testset_meds = ReadingData(path_t=test_meds_filename)#, path_l=test_demogs_filename)
                testset_diags = ReadingData(path_t=test_diags_filename)#, path_l=test_demogs_filename)
                testset_procs = ReadingData(path_t=test_procs_filename)#, path_l=test_demogs_filename)
                testset_demogs = ReadingData(path_t=test_demogs_filename)
                test_metadata = ReadingData(path_t=test_metadata_filename)
                
                test_data = testset_meds.data
                test_data_ar=np.array(test_data)
                val_med_enrolids = test_data_ar[:,0]
                test_medications=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_meds+1))  # +1 for visit date 
                
                test_data = testset_diags.data
                test_data_ar=np.array(test_data)
                val_diags_enrolids = test_data_ar[:,0]
                test_diagnoses=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_diags+1))  # +1 for visit date 

                test_data = testset_procs.data
                test_data_ar=np.array(test_data)
                val_procs_enrolids = test_data_ar[:,0]
                test_procedures=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_procs+1))  # +1 for visit date 
             

                test_data = testset_demogs.data
                test_data_ar=np.array(test_data)
                val_demogs_enrolids = test_data_ar[:,0]
                test_demographics=np.reshape(test_data_ar[:,one:],(len(test_data_ar), num_time_steps, d_demogs+1))  # +1 for visit date 
                
                labels_test = np.zeros((len(test_metadata.data), 2))
                labels_temp = np.array(test_metadata.data)[:,-1]
                
                labels_test[:,0] = labels_temp
                labels_test[labels_temp == 0, 1] = 1
                if not(sum(val_med_enrolids == val_diags_enrolids) == sum(val_med_enrolids== val_procs_enrolids) == sum(val_med_enrolids==val_demogs_enrolids) == len(val_med_enrolids)):
                    print("Error: test enrolids don't match!")
                    pdb.set_trace()

                test_meds_diags_procs_demogs=np.concatenate((test_medications[:,:,one:], test_diagnoses[:,:, one:], test_procedures[:,:,one:], test_demographics[:,:,one:]), axis=2)
                #========== Findinmg the sequence lengths
                #pdb.set_trace()
                # test_seq_length=[]
                # for i in range(len(test_meds_diags_procs_demogs)):
                #     test_seq_length.append(find_length(test_meds_diags_procs_demogs[i][:,:-d_demogs]))
                test_seq_beg=[]
                test_seq_end=[]
                for i in range(len(test_meds_diags_procs_demogs)):
                    beginning, end = find_beg_end(test_meds_diags_procs_demogs[i][:,:-d_demogs])
                    test_seq_beg.append(beginning)
                    test_seq_end.append(end)                                

                # testing_meds = ReadingData(path_t=test_meds_filename, path_l=test_labels_filename)#,path_s=testing_lengths_filename)
                # testing_diags = ReadingData(path_t=test_diags_filename, path_l=test_labels_filename)#,path_s=testing_lengths_filename)
                # testing_demogs = ReadingData(path_t=test_demogs_filename, path_l=test_labels_filename)#,path_s=testing_lengths_filename)

                # testing_data = testing_meds.data
                # testing_label = testing_meds.labels
                # testing_data_ar=np.array(testing_data)
                # testing_medications=np.reshape(testing_data_ar[:,one:],(len(testing_data_ar), num_time_steps, d_meds+1))   
                # testing_labels=np.array(testing_label)

                # testing_data = testing_diags.data
                # testing_data_ar=np.array(testing_data)
                # testing_diagnoses=np.reshape(testing_data_ar[:,one:],(len(testing_data_ar), num_time_steps, d_diags+1))   

                # testing_data = testing_demogs.data
                # testing_data_ar=np.array(testing_data)
                # testing_demographics=np.reshape(testing_data_ar[:,one:],(len(testing_data_ar), num_time_steps, d_demogs+1))   
                                
                # testing_meds_diags_demogs=np.concatenate((testing_medications[:,:,one:], testing_diagnoses[:,:, one:], testing_demographics[:,:, one:]), axis=2)
                
                # testing_seq_beg=[]
                # testing_seq_end=[]
                # for i in range(len(testing_medications)):
                #     beginning, end = find_beg_end(testing_meds_diags_demogs[i][:,:-d_demogs])
                #     testing_seq_beg.append(beginning)
                #     testing_seq_end.append(end)
                # labels_testing = testing_labels[:,one:].astype(np.float32)        
                # pdb.set_trace()                    

                #================= Representing data        
                [states_val, outputs_original_val] =sess.run([states, outputs_original], feed_dict={x: validation_meds_diags_procs_demogs, y: labels_validation, seqlen: validation_seq_end})            

                array_temp=np.array(outputs_original_val).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                # pdb.set_trace()  
                labels_idx=0              
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = labels_validation[labels_idx,0]
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, current_patient)))
                    valid_rep_file.write(',')
                    valid_rep_file.write(','.join(map(repr, labels_validation[labels_idx, one:])))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(validation_seq_beg[labels_idx])))
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(float(validation_seq_end[labels_idx])))                    
                    valid_rep_file.write(',')
                    valid_rep_file.write(str(current_patient_enrolid))
                    valid_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1   
                pdb.set_trace()
                [states_train, outputs_original_train] =sess.run([states, outputs_original], feed_dict={x: train_meds_diags_procs_demogs, y: labels_train,seqlen: train_seq_end})            
           
                array_temp=np.array(outputs_original_train).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                labels_idx=0     
                # pdb.set_trace()                         
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = labels_train[labels_idx,0]
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, current_patient)))
                    train_rep_file.write(',')
                    train_rep_file.write(','.join(map(repr, labels_train[labels_idx, one:])))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(train_seq_beg[labels_idx])))
                    train_rep_file.write(',')
                    train_rep_file.write(str(float(train_seq_end[labels_idx])))                    
                    train_rep_file.write(',')
                    train_rep_file.write(str(current_patient_enrolid))
                    train_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1                        
                # pdb.set_trace()    
                [states_test, outputs_original_test] =sess.run([states, outputs_original], feed_dict={x: test_meds_diags_procs_demogs, y: labels_test,seqlen: test_seq_end})            
           
                array_temp=np.array(outputs_original_test).flatten()
                array_temp_reshaped=np.reshape(array_temp,(num_time_steps, -1))
                start_idx=0
                labels_idx=0                              
                while start_idx < array_temp_reshaped.shape[1]:
                    slice_temp = array_temp_reshaped[:,start_idx:start_idx+n_hidden]
                    current_patient = slice_temp.flatten()
                    current_patient_enrolid = labels_test[labels_idx,0]
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, current_patient)))
                    test_rep_file.write(',')
                    test_rep_file.write(','.join(map(repr, labels_test[labels_idx, one:])))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(test_seq_beg[labels_idx])))
                    test_rep_file.write(',')
                    test_rep_file.write(str(float(test_seq_end[labels_idx])))                    
                    test_rep_file.write(',')
                    test_rep_file.write(str(current_patient_enrolid))
                    test_rep_file.write('\n')
                    start_idx += n_hidden 
                    labels_idx +=1           
                pdb.set_trace()                 
    return softmax_predictions_temp, accuracy, precision, recall, sensitivity, specificity, tp, tn, fp, fn, validation_auc#, accuracy_temp_train, precision_train, recall_train, sensitivity_train, specificity_train,tp_train, tn_train, fp_train, fn_train

if __name__ == "__main__": main(idx,representing, epochs, reg_coeff, learning_rt, n_hid, batch_sz, train_meds_filename, train_diags_filename, train_procs_filename, train_demogs_filename, train_metadata_filename, validation_meds_filename, validation_diags_filename, validation_procs_filename, validation_demogs_filename, validation_metadata_filename)


