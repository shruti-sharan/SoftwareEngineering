import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import pickle


pos_embedding = pickle.load(open('pos_embedding','rb'))


neg_embedding = pickle.load(open('neg_embedding','rb'))
neg_embedding=neg_embedding[:6291]
len(neg_embedding)


# In[7]:


#concatenating
all_embeddings=list(pos_embedding+neg_embedding)

pos_target=np.ones(6291)

neg_target=np.zeros(6291)


all_target= np.append(pos_target,neg_target,axis=None)



embedding_size = 361
question_size = 104
def get_padded_embeddings(temp_df):
    #print("tdf", temp_df.shape)
    temp_df=temp_df.reshape(-1,361)
    zero_embeddings = np.zeros(embedding_size)
    truncated_df = temp_df[:question_size]
    #print(truncated_df.shape)
    #print(len(truncated_df),len(truncated_df[0]))
    #print(type([zero_embeddings]*(question_size-len(truncated_df))))
    #print(np.array([zero_embeddings]*(question_size - len(truncated_df))).shape)
    if(len(truncated_df)!=104):
        truncated_df = np.concatenate((truncated_df, np.array([zero_embeddings]*(question_size- len(truncated_df)))))
        #print(truncated_df.shape)
    return truncated_df


def batch_gen(n_batches,y,all_embeddings):
    #print("bg")
    while True: 
        for i in range(n_batches):
            #print("for loop")
            embeddings_list = all_embeddings[i*batch_size:(i+1)*batch_size] 
            #print("el",embeddings_list.shape,all_embeddings.shape,embeddings_list[0].shape)
            concat_embed_ques = np.array([get_padded_embeddings(ques) for ques in all_embeddings])
            #print(i,concat_embed_ques.shape)
            yield concat_embed_ques, np.array(y[i*batch_size:(i+1)*batch_size])



emb_vect_temp=np.array([get_padded_embeddings(line) for line in all_embeddings])



all_embeddings_train, all_embeddings_test,y_train,y_test = train_test_split(emb_vect_temp, all_target, test_size = 0.30, shuffle=True, random_state = 42)


batch_size=32
n_batches = math.ceil(len(y_train)/batch_size)
bg=batch_gen(n_batches,y_train,all_embeddings_train)



from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional
from sklearn.metrics import f1_score

model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(104, 361)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history=model.fit_generator(bg, epochs=2,
                    steps_per_epoch=1000,
                    verbose=True)

# In[ ]:


with open('HistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# In[ ]:


with open('Model', 'wb') as file_mo:
        pickle.dump(model, file_mo)




test_gen=batch_gen(n_batches,y_test,all_embeddings_test)
scores=model.evaluate_generator(test_gen,steps=100,verbose=1)
print("Accuracy", scores[1])
pred_val=model.predict_generator(test_gen, steps=100,verbose=1)
print("Predicted val", pred_val[1])



model.summary()

plot_history(history)