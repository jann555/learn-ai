#!/usr/bin/env python
# coding: utf-8

# # Embedding functions
# 
# Using the appropriate model, anything can be embedded. For this exercise we're going to use embeddings to do semantic search over Rick and Morty quotes

# ### ChatGPT doesn't have parents or emotions
# 
# Vector search is useful for retrieving data that's not part of the model's training data.
# 
# For example, if we asked the following question to ChatGPT, we get some generic sounding answer wrapping around the core "she does not make any statements about causing her parents misery".

# ![image.png](attachment:image.png)

# But what if searched through actual quotes from the show?

# ### [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
# 
# Restart the kernel (`Kernel` --> `Restart` in the Jupyter Notebook menu) after running this cell to use the latest packages.

# In[1]:


get_ipython().system('pip install -U --quiet sentence-transformers==2.5.1 transformers==4.36.0')


# ### Read data
# 
# Now let's read the quotes from the included text file (source: https://parade.com/tv/rick-and-morty-quotes). 

# In[3]:


def read_quotes() -> list[str]:
    with open('rick_and_morty_quotes.txt', "r") as fh:
        return fh.readlines()


# In[4]:


rick_and_morty_quotes = read_quotes()
rick_and_morty_quotes[:3]


# Oops it seems like we have some extra newlines at the end of these quotes. Does that matter?
# How do we prove it to ourselves?

# In[7]:


import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

emb1, emb2 = model.encode(["Losers look stuff up while the rest of us are carpin' all them diems.\n",
                          "Losers look stuff up while the rest of us are carpin' all them diems."])

np.allclose(emb1, emb2)


# ### Write function to generate embeddings from text
# 
# Write a function that turns text into embeddings using Sentence Transformers.
# 
# **HINT**
# 1. Choose a [pre-trained model](https://www.sbert.net/docs/pretrained_models.html), you don't need to create your own
# 2. See the API documentation and examples for Sentence Transformers to see how to encode text

# In[11]:


import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:  
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(input_data)


# In[12]:


embeddings = generate_embeddings(rick_and_morty_quotes)


# In[13]:


#Print the embeddings
for sentence, embedding in zip(rick_and_morty_quotes[:3], embeddings[:3]):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")


# https://www.sbert.net/docs/pretrained_models.html
# 

# How many dimensions is each embedding?

# In[14]:


len(embeddings[0])


# Are the embeddings normalized already?
# 

# In[15]:


np.linalg.norm(embeddings,axis=1)


# ### Let's put it all together
# 
# First let's encode the question

# In[16]:


query_text = "Are you the cause of your parents' misery?"
query_embedding = model.encode(query_text)


# Now we can reuse the find_nearest_neighbors function we wrote for exercise 1.
# 
# However, that only returns the vectors, whereas we also want the quotes. So please rewrite the find_nearest_neighbors function to return the *indices* of the nearest neighbors.

# In[22]:


import numpy as np

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape)-1)


def find_nearest_neighbors(query: np.ndarray,
                           vectors: np.ndarray,
                           k: int = 1) -> np.ndarray:
    """
    Find k-nearest neighbors of a query vector.

    Parameters
    ----------
    query : np.ndarray
        Query vector.
    vectors : np.ndarray
        Vectors to search.
    k : int, optional
        Number of nearest neighbors to return, by default 1.

    Returns
    -------
    np.ndarray
        The indices of the `k` nearest neighbors of `query` in `vectors`.
    """
    distances = euclidean_distance(query, vectors)
    return np.argsort(distances)[:k]


# In[23]:


indices = find_nearest_neighbors(query_embedding, embeddings, k=3)


# In[24]:


for i in indices:
    print(rick_and_morty_quotes[i])


# #### Asking the question again
# 
# Now let's use the retrieved quotes and ask ChatGPT to answer the question based on the quotes in addition to its own data

# In[25]:


"""
Answer the question based on the context.

Question: Are you the cause of your parents' misery?

Context:

You're not the cause of your parents' misery. You're just a symptom of it.

Having a family doesn't mean that you stop being an individual. You know the best thing you can do for the people that depend on you? Be honest with them, even if it means setting them free.

B—h, my generation gets traumatized for breakfast.
"""


# ![image.png](attachment:image.png)

# In[ ]:




