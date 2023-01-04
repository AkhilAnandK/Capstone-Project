Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
= RESTART: C:/Users/kundu/AppData/Local/Programs/Python/Python38/capstone_trial_2.py
Some layers from the model checkpoint at bert-base-uncased were not used when initializing TFBertModel: ['mlm___cls', 'nsp___cls']
- This IS expected if you are initializing TFBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
All the layers of TFBertModel were initialized from the model checkpoint at bert-base-uncased.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_ids (InputLayer)         [(None, 300)]        0           []                               
                                                                                                  
 attention_masks (InputLayer)   [(None, 300)]        0           []                               
                                                                                                  
 token_type_ids (InputLayer)    [(None, 300)]        0           []                               
                                                                                                  
 bert (TFBertMainLayer)         TFBaseModelOutputWi  109482240   ['input_ids[0][0]',              
                                thPoolingAndCrossAt               'attention_masks[0][0]',        
                                tentions(last_hidde               'token_type_ids[0][0]']         
                                n_state=(None, 300,                                               
                                 768),                                                            
                                 pooler_output=(Non                                               
                                e, 768),                                                          
                                 past_key_values=No                                               
                                ne, hidden_states=N                                               
                                one, attentions=Non                                               
                                e, cross_attentions                                               
                                =None)                                                            
                                                                                                  
 bidirectional (Bidirectional)  (None, 300, 128)     426496      ['bert[0][0]']                   
                                                                                                  
 global_average_pooling1d (Glob  (None, 128)         0           ['bidirectional[0][0]']          
 alAveragePooling1D)                                                                              
                                                                                                  
 global_max_pooling1d (GlobalMa  (None, 128)         0           ['bidirectional[0][0]']          
 xPooling1D)                                                                                      
                                                                                                  
 concatenate (Concatenate)      (None, 256)          0           ['global_average_pooling1d[0][0]'
                                                                 , 'global_max_pooling1d[0][0]']  
                                                                                                  
 dropout_37 (Dropout)           (None, 256)          0           ['concatenate[0][0]']            
                                                                                                  
 dense (Dense)                  (None, 1)            257         ['dropout_37[0][0]']             
                                                                                                  
==================================================================================================
Total params: 109,908,993
Trainable params: 426,753
Non-trainable params: 109,482,240
__________________________________________________________________________________________________
Epoch 1/15

Epoch 2/15

Epoch 3/15

Epoch 4/15

Epoch 5/15

Epoch 6/15

Epoch 7/15

Epoch 8/15

Epoch 9/15

Epoch 10/15

Epoch 11/15

Epoch 12/15

Epoch 13/15

Epoch 14/15

Epoch 15/15

>>> test_data = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
>>> a=BertSemanticDataGenerator.model.evaluate(test_data,verbose=1)

>>> predictions=BertSemanticDataGenerator.model.predict(test_data)
>>> predictions.size
1376
>>> test_data_1 = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=30,
    shuffle=False,
)
>>> b=BertSemanticDataGenerator.model.evaluate(test_data_1,verbose=1)

>>> predictions_1=BertSemanticDataGenerator.model.predict(test_data_1)
>>> predictions_1.size
1350
>>> import pandas
>>> output_df=pandas.DataFrame(predictions)
>>> output_df.to_excel('new_predictions.xlsx', index=False)
>>> 