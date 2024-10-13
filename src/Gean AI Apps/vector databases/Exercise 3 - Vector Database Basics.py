#!/usr/bin/env python
# coding: utf-8
# import timeit

# # Vector Database Basics
# 
# Vector databases help us store, manage, and query the embeddings we created for generative AI, recommenders,
# and search engines.
# 
# Across many of the common use cases, users often find that they need to manage more than just vectors.
# To make it easier for practitioners, vector databases should store and manage all of the data they need:
# - embedding vectors
# - categorical metadata
# - numerical metadata
# - timeseries metadata
# - text / pdf / images / video / point clouds
# 
# And support a wide range of query workloads:
# - Vector search (may require ANN-index)
# - Keyword search (requires full text search index)
# - SQL (for filtering)
# 
# For this exercise we'll use LanceDB since it's open source and easy to setup

# In[ ]:


# pip install -U --quiet lancedb pandas pydantic [This has been pre-installed for you]


# ## Creating tables and adding data
# 
# Let's create a LanceDB table called `cats_and_dogs` under the local database directory `~/.lancedb`.
# This table should have 4 fields:
# - the embedding vector
# - a string field indicating the species (either "cat" or "dog")
# - the breed
# - average weight in pounds

# We're going to use pydantic to make this easier. First let's create a pydantic model with those fields

# In[47]:


from lancedb.pydantic import vector, LanceModel

class CatsAndDogs(LanceModel):
    vector: vector(2)
    species: str
    breed: str
    weight: float


# Now connect to a local db at ~/.lancedb and create an empty LanceDB table called "cats_and_dogs"

# In[48]:


import lancedb

db = lancedb.connect("~/.lancedb")
table = db.create_table("cats_and_dogs", schema=CatsAndDogs, mode="create")


# Let's add some data

# First some cats

# In[49]:


data = [
    CatsAndDogs(
        vector=[1., 0.],
        species="cat",
        breed="shorthair",
        weight=12.,
    ),
    CatsAndDogs(
        vector=[-1., 0.],
        species="cat",
        breed="himalayan",
        weight=9.5,
    ),
]


# Now call the `LanceTable.add` API to insert these two records into the table

# In[50]:


table.add([dict(d) for d in data])


# Let's preview the data

# In[51]:


table.head().to_pandas()


# Now let's add some dogs

# In[52]:


data = [
    CatsAndDogs(
        vector=[0., 10.],
        species="dog",
        breed="samoyed",
        weight=47.5,
    ),
    CatsAndDogs(
        vector=[0, -1.],
        species="dog",
        breed="corgi",
        weight=26.,
    )
]


# In[53]:


table.add([dict(d) for d in data])


# In[54]:


table.head().to_pandas()


# ## Querying tables
# 
# Vector databases allow us to retrieve data for generative AI applications. Let's see how that's done.

# Let's say we have a new animal that has embedding [10.5, 10.], what would you expect the most similar animal will be?
# Can you use the table we created above to answer the question?
# 
# **HINT** you'll need to use the `search` API for LanceTable and `limit` / `to_df` APIs. For examples you can refer to [LanceDB documentation](https://lancedb.github.io/lancedb/basic/#how-to-search-for-approximate-nearest-neighbors).

# In[55]:


table.search([10.5, 10.]).limit(1).to_df()


# Now what if we use cosine distance instead? Would you expect that we get the same answer? Why or why not?
# 
# **HINT** you can add a call to `metric` in the call chain

# In[56]:


table.search([10.5, 10.]).metric("cosine").limit(1).to_df()


# ## Filtering tables
# 
# In practice, we often need to specify more than just a search vector for good quality retrieval. Oftentimes we need to filter the metadata as well.

# Please write code to retrieve two most similar examples to the embedding [10.5, 10.] but only show the results that is a cat.

# In[61]:


table.search([10.5, 10.,]).limit(4).where("species = 'cat'",).to_df()


# ## Creating ANN indices
# 
# For larger tables (e.g., >1M rows), searching through all of the vectors becomes quite slow. Here is where the Approximate Nearest Neighbor (ANN) index comes into play. While there are many different ANN indexing algorithms, they all have the same purpose - to drastically limit the search space as much as possible while losing as little accuracy as possible
# 
# For this problem we will create an ANN index on a LanceDB table and see how that impacts performance

# ### First let's create some data
# 
# Given the constraints of the classroom workspace, we'll complete this exercise by creating 100,000 vectors with 16D in a new table. Here the embedding values don't matter, so we simply generate random embeddings as a 2D numpy array. We then use the vec_to_table function to convert that in to an Arrow table, which can then be added to the table.

# In[62]:


from lance.vector import vec_to_table
import numpy as np

mat = np.random.randn(100_000, 16)
table_name = "exercise3_ann"
db.drop_table(table_name, ignore_missing=True)
table = db.create_table(table_name, vec_to_table(mat))


# ### Let's establish a baseline without an index
# 
# Before we create the index, let's make sure know what we need to compare against.
# 
# We'll generate a random query vector and record it's value in the `query` variable so we can use the same query vector with and without the ANN index.

# In[63]:


query = np.random.randn(16)
table.search(query).limit(10).to_df()


# Please write code to compute the average latency of this query

# In[64]:


# timeit table.search(query).limit(10).to_arrow()


# ### Now let's create an index
# 
# There are many possible index types ranging from hash based to tree based to partition based to graph based.
# For this task, we'll create an IVFPQ index (partition-based index with product quantization compression) using LanceDB.
# 
# Please create an IVFPQ index on the LanceDB table such that each partition is 4000 rows and each PQ subvector is 8D.
# 
# **HINT** 
# 1. Total vectors / number of partitions = number of vectors in each partition
# 2. Total dimensions / number of subvectors = number of dimensions in each subvector
# 3. This step can take about 7-10 minutes to process and execute in the classroom workspace.

# In[ ]:


table.create_index(num_partitions=256, num_sub_vectors=16)


# Now let's search through the data again. Notice how the answers now appear different.
# This is because an ANN index is always a tradeoff between latency and accuracy.

# In[ ]:


table.search(query).limit(10).to_df()


# Now write code to compute the average latency for querying the same table using the ANN index.
# 
# **SOLUTION** The index is implementation detail, so it should just be running the same code as above. You should see almost an order of magnitude speed-up. On larger datasets, this performance difference should be even more pronounced.

# In[ ]:


# timeit table.search(query).limit(10).to_arrow()


# ## Deleting rows
# 
# Like with other kinds of databases, you should be able to remove rows from the table.
# Let's go back to our tables of cats and dogs

# In[ ]:


table = db["cats_and_dogs"]


# In[ ]:


len(table)


# Can you use the `delete` API to remove all of the cats from the table?
# 
# **HINT** use a SQL like filter string to specify which rows to delete from the table

# In[ ]:


table.delete("species = 'cat'")


# In[ ]:


len(table)


# ## What if I messed up?
# 
# Errors is a common occurrence in AI. What's hard about errors in vector search is that oftentimes a bad vector doesn't cause a crash but just creates non-sensical answers. So to be able to rollback the state of the database is very important for debugging and reproducibility

# So far we've accumulated 4 actions on the table:
# 1. creation of the table
# 2. added cats
# 3. added dogs
# 4. deleted cats

# What if you realized that you should have deleted the dogs instead of the cats?
# 
# Here we can see the 4 versions that correspond to the 4 actions we've done

# In[ ]:


table.list_versions()


# Please write code to restore the version still containing the whole dataset

# In[ ]:


table = db["cats_and_dogs"]


# In[ ]:


len(table)


# In[ ]:


# restore to version 3

table.restore(3)


# In[ ]:


# delete the dogs instead

table.delete("species = 'dog'")


# In[ ]:


table.list_versions()


# In[ ]:


table.to_pandas()


# ## Dropping a table
# 
# You can also choose to drop a table, which also completely removes the data.
# Note that this operation is not reversible.

# In[ ]:


"cats_and_dogs" in db


# Write code to irrevocably remove the table "cats_and_dogs" from the database

# In[ ]:


table.drop("cats_and_dogs" )


# How would you verify that the table has indeed been deleted?

# In[ ]:


table.name in db


# ## Summary
# 
# Congrats, in this exercise you've learned the basic operations of vector databases from creating tables, to adding data, and to querying the data. You've learned how to create indices and you saw first hand how it changes the performance and the accuracy. Lastly, you've learned how to debug and rollback when errors happen.

# In[ ]:




