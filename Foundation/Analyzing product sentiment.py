
# coding: utf-8

# #Predicting sentiment from product reviews
# 
# #Fire up GraphLab Create

# In[1]:

import graphlab


# #Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[3]:

#products = graphlab.SFrame('amazon_baby.gl/')
products=graphlab.SFrame('amazon_baby.gl/')


# #Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[4]:

#products.head()
products.head()


# #Build the word count vector for each review

# In[5]:

#products['word_count'] = graphlab.text_analytics.count_words(products['review'])
products['word_count']= graphlab.text_analytics.count_words(products['review'])


# In[5]:

products.head()


# In[10]:

#graphlab.canvas.set_target('ipynb')
graphlab.canvas.set_target('browser')


# In[11]:

products['name'].show()


# #Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[13]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[14]:

len(giraffe_reviews)


# In[15]:

giraffe_reviews['rating'].show(view='Categorical')


# #Build a sentiment classifier

# In[16]:

products['rating'].show(view='Categorical')


# ##Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[17]:

#ignore all 3* reviews
products = products[products['rating'] != 3]


# In[18]:

#positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[20]:

products.show()


# ##Let's train the sentiment classifier

# In[22]:

train_data,test_data = products.random_split(.8, seed=0)


# In[25]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# #Evaluate the sentiment model

# In[26]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[29]:

sentiment_model.show(view='Evaluation')


# #Applying the learned model to understand sentiment for Giraffe

# In[30]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[31]:

giraffe_reviews.head()


# ##Sort the reviews based on the predicted sentiment and explore

# In[32]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[33]:

giraffe_reviews.head()


# ##Most positive reviews for the giraffe

# In[34]:

giraffe_reviews[0]['review']


# In[35]:

giraffe_reviews[1]['review']


# ##Show most negative reviews for giraffe

# In[36]:

giraffe_reviews[-1]['review']


# In[39]:

giraffe_reviews[-2]['review']


# In[38]:

giraffe_reviews[-2]['word_count']


# In[ ]:



