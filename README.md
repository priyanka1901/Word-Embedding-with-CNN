
# Word-Embedding-with-CNN

![image](https://github.com/priyanka1901/Word-Embedding-with-CNN/blob/master/text_class.png)

# Document classification

Document classification is an example of Machine learning where we classify text based on its content.

There are two broad categories of Machine learning techniques which can be used for it.

# Supervised leaning — 
Where we already have the category to which particular document belongs to. Our model parse through the data during training, maps the function from it.
Categories are predefined and documents within the training datasets are manually tagged with one or more category labels.After training, the model is smart enough to categorize the new document given.
# Unsupervised learning —
Where we do not have the class label attached to the document and we use ML algorithms to cluster the document which are of same type.

Refer below diagram for better understanding-

![document](https://github.com/priyanka1901/Word-Embedding-with-CNN/blob/master/text-analysis-acme2.jpg)

# Preprocessing

Lets suppose we have millions of emails with us and we need to classify to which class each of these email belongs to.

In real world the data given is never perfect. We need to do preprocessing so as to extract maximum knowledge out of it with out making our model get confused due to extra information given .

Take out the subject, remove extra details from it and put it in a Data-frame.

Extract all the Email ids mentioned into the mail and get it into a Data-frame.

Extract the given text data , preprocess it and put it in a Data-frame.

Combine all these and we are ready with the desired text to give to our model.

# Word embedding

Word Embedding is a representation of text where words that have the same meaning have a similar representation. In other words it represents words in a coordinate system where related words, based on a corpus of relationships, are placed closer together.

It is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to dense vector of real numbers.

It is actually an improvement over traditional ways of encoding such as Bag-of-word where each word was represented my a large sparse vector depending upon size of vocabulary it is dealing with.

In contrast to this, in an embedding, representation of a word is by a dense vector which represents the projection of the word into a continuous vector space.

The position of a word within the vector space is learned from text and is based on the neighboring words.

# Keras Embedding layer

Keras offers an Embedding layer that can be used for neural networks on text data. It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with keras.

This layer can be represented as —

![keras](https://github.com/priyanka1901/Word-Embedding-with-CNN/blob/master/keras.png)

Where few important arguments are-

# input_dim — 
Which specifies the size of the vocabulary in the text data, that means the total unique words in your data is 50.
# output_dim — 
Which specifies the size of the word vector you get as the output of the embedding layer.
# input_length — 
It is the length of the input sequence. For example if the maximum length of a sentence in a document is 100 then its input length is 100.
# Trainable — 
It specifies whether we want to train the embedding layer or not.

The embedding layer can be used in different ways-

We use it as a part of deep learning model and this layer learns with the model itself. In such scenarios we give parameter trainable as True.

We use already pretrained vectors to represent our words which are trained on large datasets.In such scenarios we give parameter trainable as False.

We will we focusing on how to use the pretrained vector for representing our words and train our complete dataset on it.

Let us take an example to understand it more deeply-

suppose we have our dataset which contains few remarks and we need to classify to which class these remarks belong to.

1 signifies that the remark is good where as 0 signifies it to be bad.

![dataset](https://github.com/priyanka1901/Word-Embedding-with-CNN/blob/master/keras1.png)

The Keras deep learning library provides some basic tools to help us prepare our text data.Text data must be encoded as numbers to be used as input or output for machine learning and deep learning models.

For this purpose we use Tokenizer API.

For more information on this, visit my blog -  https://medium.com/@gunjan.priyanka29/classification-of-documents-using-convolutional-neural-network-cnn-e0768bb81aad
