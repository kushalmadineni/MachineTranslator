import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense

# Load dataset
def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    english_sentences = []
    hindi_sentences = []
    for line in lines:
        # Check if the line contains at least one tab
        if '\t' in line:
            english, hindi = line.strip().split('\t')
            english_sentences.append(english)
            hindi_sentences.append(hindi)
    return english_sentences, hindi_sentences

# Load data from hindi.txt
english_sentences, hindi_sentences = load_dataset('hindi.txt')

if not english_sentences or not hindi_sentences:
    print("Error: No data loaded from the dataset.")
    exit()

# Tokenize English and Hindi sentences
english_tokenizer = Tokenizer()
english_tokenizer.fit_on_texts(english_sentences)
english_tokenizer.word_index['<start>'] = len(english_tokenizer.word_index) + 1
english_tokenizer.word_index['<end>'] = len(english_tokenizer.word_index) + 1

hindi_tokenizer = Tokenizer()
hindi_tokenizer.fit_on_texts(hindi_sentences)
hindi_tokenizer.word_index['<start>'] = len(hindi_tokenizer.word_index) + 1
hindi_tokenizer.word_index['<end>'] = len(hindi_tokenizer.word_index) + 1

# Pad sequences
max_length = max(len(seq.split()) for seq in english_sentences)
max_length_hindi = max(len(seq.split()) for seq in hindi_sentences)

if max_length == 0 or max_length_hindi == 0:
    print("Error: Empty sequences detected in the dataset.")
    exit()

# Build LSTM model
vocab_size_english = len(english_tokenizer.word_index) + 1
vocab_size_hindi = len(hindi_tokenizer.word_index) + 1
embedding_dim = 256
units = 512

# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = Embedding(vocab_size_english, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = Embedding(vocab_size_hindi, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_hindi, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([pad_sequences(english_tokenizer.texts_to_sequences(english_sentences), maxlen=max_length, padding='post'),
           pad_sequences(hindi_tokenizer.texts_to_sequences(hindi_sentences), maxlen=max_length_hindi, padding='post')],
          np.expand_dims(pad_sequences(hindi_tokenizer.texts_to_sequences(hindi_sentences), maxlen=max_length_hindi, padding='post'), -1), epochs=10)

# Inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(units,))
decoder_state_input_c = tf.keras.layers.Input(shape=(units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Translate an English sentence to Hindi
def translate_sentence(input_sentence):
    input_sequence = english_tokenizer.texts_to_sequences([input_sentence])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    
    states_value = encoder_model.predict(input_sequence_padded)
    
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = hindi_tokenizer.word_index['<start>']
    
    stop_condition = False
    translated_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_sequence] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        
        for word, index in hindi_tokenizer.word_index.items():
            if index == sampled_token_index:
                translated_sentence += ' {}'.format(word)
                sampled_word = word
                break
                
        if sampled_word == '<end>' or len(translated_sentence.split()) > max_length_hindi:
            stop_condition = True
            
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = sampled_token_index
        
        states_value = [h, c]
        
    return translated_sentence.strip()

# Get input from user and translate
input_sentence = input("Enter an English sentence: ")
translated_sentence = translate_sentence(input_sentence)
print("Hindi Translation:", translated_sentence)
