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
print(len(pos_embedding))



pos_embedding[1].shape
print(type(pos_embedding[0]))



neg_embedding = pickle.load(open('neg_embedding','rb'))
print(len(neg_embedding))



#sub smaple negative samples
neg_embedding=neg_embedding[:6291]
print(len(neg_embedding))





#concatenating
all_embeddings=list(pos_embedding+neg_embedding)
print(len(all_embeddings))



#creating two arrays with 1s and 0s
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


# In[17]:


emb_vect_temp=np.array([get_padded_embeddings(line) for line in all_embeddings])


all_embeddings_train, all_embeddings_test,y_train,y_test = train_test_split(emb_vect_temp, all_target, test_size = 0.30, shuffle=True, random_state = 42)

emb_vect=np.array([get_padded_embeddings(line) for line in all_embeddings_train])



from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional,Flatten
from sklearn.metrics import f1_score
from keras.utils import to_categorical


# In[29]:


model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(104,361))))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation="sigmoid"))


# In[30]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


history=model.fit(all_embeddings_train, y_train, batch_size=None,nb_epoch=1,steps_per_epoch=1000, verbose=1)


# In[25]:
with open('HistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

with open('Model', 'wb') as file_mo:
        pickle.dump(model, file_mo)



model.summary()

plot_history(history)

score=model.evaluate(all_embeddings_test,y_test,steps_per_epoch=1000,verbose=1)
print("Test Accuracy",score[1])

# In[ ]:




