
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target

