import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from tensorflow.keras.models import load_model
import tensorflow.keras.utils

# Maximum length of input sentence to the model.
max_length = 300
batch_size = 32
epochs = 15

# Loading the dataset
train_df=pd.read_csv("train_main.csv")
valid_df=pd.read_csv("dev_main.csv")
test_df=pd.read_csv("test_main.csv")
y_train=train_df.score
y_val=valid_df.score
y_test=test_df.score
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
        ):
            self.sentence_pairs = sentence_pairs
            self.labels = labels
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.include_targets = include_targets
            # Load our BERT Tokenizer to encode the text.
            # We will use base-base-uncased pretrained model.
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
            )
            self.indexes = np.arange(len(self.sentence_pairs))
            self.on_epoch_end()

    def __len__(self):
            # Denotes the number of batches per epoch.
            return len(self.sentence_pairs) // self.batch_size
        
    def __getitem__(self, idx):
            # Retrieves the batch of index.
            indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
            sentence_pairs = self.sentence_pairs[indexes]
    
        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
            encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="tf",
                )
            # Convert batch of encoded features to numpy array.
            input_ids = np.array(encoded["input_ids"], dtype="int32")
            attention_masks = np.array(encoded["attention_mask"], dtype="int32")
            token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
            # Set to true if data generator is used for training/validation.
            if self.include_targets:
                labels = np.array(self.labels[indexes], dtype="float64")
                return [input_ids, attention_masks, token_type_ids], labels
            else:
                return [input_ids, attention_masks, token_type_ids]
        
    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
            if self.shuffle:
                np.random.RandomState(42).shuffle(self.indexes)

    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        #Loading the bertmodel
    from transformers import BertTokenizer, BertModel
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False
    bert_output = bert_model.bert(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
            )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(1)(dropout)
    model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
            )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss="mse",
        metrics=["mean_squared_error"],
    )
    model.summary()
    
#Encoding the data
train_data = BertSemanticDataGenerator(
    train_df[["sentence1", "sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_df[["sentence1", "sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)
#Training
print("\nTraining has started")
history = BertSemanticDataGenerator.model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
print("\nTraining has completed")
BertSemanticDataGenerator.model.save("Capstone_model.h5")
print("\nThe trained model is saved with the name Capstone_model.h5")
#Testing
print("\nTesting\n")
test_data = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=True,
)
#Evaluation on Test data
a=BertSemanticDataGenerator.model.evaluate(test_data,verbose=1)
print("\nLoss on Testing data is: ",a[0])
print("\nMean Squared Error on testing data is: ",a[1])
print("\nThe testing has been completed")
#Predictions
print("\nThe model is making the predictions now. Please wait\n")
predictions = BertSemanticDataGenerator.model.predict(test_data)
print("Predictions are: ",predictions)
output_df=pd.DataFrame(predictions)
output_df.to_excel('predictions.xlsx', index=False)
print("\n The execution has completed. Kindly check the predictions.xlsx file in the folder")
        
