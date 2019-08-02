---
layout: post
title:  How to build a simple hate speech detector with machine learning
category: jekyll 
date: 2019-08-2 15:00:00 +0200
categories: nlp
image: "/assets/posts/post-no-hate.jpg"
---

Not everybody on the internet behaves nice and some comments are just rude or offending. If you run a web page that offers a public comment function hate speech can be a real problem. For example in Germany, you are legally required to delete hate speech comments. This can be challenging if you have to check thousands of comments each day. 
So wouldn't it be nice, if you can automatically check the user's comment and give them a little hint to stay nice?
<!--description-->

The simplest thing you could do is to check if the user's text contains offensive words. However, this approach is limited since you can offend people without using offensive words. 

This post will show you how to train a machine learning model that can detect if a comment or text is offensive. And to start you need just a few lines of Python code \o/

## The Data

At first, you need data. In this case, you will need a list of offensive and nonoffensive texts. I wrote this tutorial for a machine learning course in Germany, so I used german texts but you should be able to use other languages too.

For a machine learning competition, scientists provided a list of comments labeled as offensive and nonoffensive ([Germeval 2018, Subtask 1](https://projects.fzai.h-da.de/iggsa/projekt/)). This is perfect for us since we just can use this data.

## The Code

To tackle this task I would first establish a baseline and then improve this solution step by step. Luckily they also published the scores of all submission, so we can get a sense of how well we are doing.

For our baseline model we are going to use [Facebooks fastText](https://fasttext.cc/). It's simple to use, works with many languages and does not require any special hardware like a GPU. Oh, and it's fast :) 

### 1. Load the data 

After you downloaded the training data file [germeval2018.training.txt](https://github.com/uds-lsv/GermEval-2018-Data) you need to transform this data into a format that fastText can read.
FastTexts standard format looks like this "__label__[your label] some text":

```
__label__offensive some insults
__label__other have a nice day
```

### 2. Train the Model

To train the model you need to install the fastText Python package.

```bash
$ pip install fasttext
```
To train the model you need just there line of code. 
```python
import fasttext
traning_parameters = {'epoch': 50, 'lr': 0.05, 'loss': "ns", 'thread': 8, 'ws': 5, 'dim': 100}    
model = fasttext.supervised('fasttext.train', 'model', **traning_parameters)
```

I packed all the training parameters into a seperate dictionary. To me that looks a bit cleaner but you don't need to do that.

### 3. Test your Model

After we trained the model it is time to test how it performs. FastText provides us a handy test method the evaluate the model's performance. To compare our model with the other models from the GermEval contest I also added a lambda which calculates the average [F1 score](https://en.wikipedia.org/wiki/F1_score). For now, I did not use the official test script from the contests repository. Which you should do if you wanted to attend to such contests.

```python
def test(model):
    f1_score = lambda precision, recall: 2 * ((precision * recall) / (precision + recall))
    nexamples, recall, precision = model.test('fasttext.test')
    print (f'recall: {recall}' )
    print (f'precision: {precision}')
    print (f'f1 score: {f1_score(precision,recall)}')
    print (f'number of examples: {nexamples}')
```

I don't know about you, but I am so curious how we score. Annnnnnnd:

```
recall: 0.7018686296715742
precision: 0.7018686296715742
f1 score: 0.7018686296715742
number of examples: 3532
```

Looking at the [results](https://github.com/uds-lsv/GermEval-2018-Data/blob/master/results.pdf) we can see that the best other model had an average F1 score of 76,77 and **our model achieves -without any optimization and preprocessing- an F1 Score of 70.18.**

This is pretty good since the models for these contests are usually specially optimized for the given data.

FastText is a clever piece of software, that uses some neat tricks. If interested in fastText you should take a look the [paper](https://arxiv.org/abs/1607.04606) and [this one](https://arxiv.org/abs/1607.01759). For example, fastText uses character n-grams. This approach is well suited for the german language, which uses a lot of compound words.

## Next Steps

In this very basic tutorial, we trained a model with just a few lines of Python code. Their several things you can do to improve this model.
The first step would be to preprocess your data. During preprocessing you could lower case all texts, remove URLs and special characters, correct spelling, etc. After every optimization step, you can test your model and check if your scores went up. Happy hacking :) 

Some Ideas:

1. Preprocess the data
2. Optimize the parameters (number of training epochs, learning rate, embedding dims, word n-grams)
3. Use pre-trained word vectors from the fastText website
4. add more data to the training set
5. Use data augmentation.

Here is the full code:

<script src="https://gist.github.com/oliverguhr/31a1c93a1005d7e6e04c23d389d89cb7.js"></script>


Credit: Photo by [Jon Tyson on Unsplash](https://unsplash.com/photos/IYtVtgXw72M)