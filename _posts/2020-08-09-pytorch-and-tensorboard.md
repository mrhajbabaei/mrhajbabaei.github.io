---
layout: post
title: Using Tensorboard in PyTorch
---

Watching closely parameters in neural networks is very important in the training process. Tensorboard recognise as a great tool for this job, but some people use think this tool is designed jsut for Tensorflow framework. However, I will show you on this post that Tensorboard can be used in PyTorch as visualization tool as well.

In this tutorial I used [an official example](https://github.com/pytorch/examples/tree/master/vae) of PyTorch about VAE network, and also final code is available [here](https://github.com/mrhajbabaei/pytorch-tensorboard) (just be aware that I changed this example to be match with our expectations). At the first step we should import **SummaryWriter**; which plays the main role in creating a summay, and I I will show you how we can use this to create different types of visualization in our program. After that, you should create an instance of **SummaryWriter**, so in top of your program in import section put this code:   
`from torch.utils.tensorboard import SummaryWriter`    
, and in your main function, or wherever you want to use that create an instance of it:    
`writer = SummaryWriter('your_log_directory/name_of_your_summary_experiment')`

**Note:** you should create your log dir already; consider something like this: "log" as your main directory for all kinds of summarization and experiment.    
**Note:** Just make sure log dire is excluded from your git repository (because of preventing upload unwanted log files on your repository).

Generally, you can add different kinds of summary for your project. In this project I consider these summary types:
#### 1- Image grid   
As I mentioned before, this project is a VAE example and input for this project is [MNIST](http://yann.lecun.com/exdb/mnist/) database which cosist of 60,000, and 10,000 images with 28x28 size for training and testing respectively. For doing so, you should make a grid by your batch data and add this image grid to your summary writer:   
```python
image_grid = make_grid(your_data_batch)  
summary_writer.add_image('mnist_images', image_grid)
```
Please consider that your data batch could be every batch data of your traning data, and the result is look like this:

![image grid](images/second-post/image_grid.png)

#### 2- Graph  
You can have a full graph of entire your network. Just by adding this snippet you can add your graph to your summary:
```python
summary_writer.add_graph(model, your_data_batch)
```
Like above, you can use every data batch you would like, and the output would be like this:

![graph](images/second-post/graph.png)

![graph](images/second-post/graph_zoom.png)

#### 3- Embedding

We use VAE in this project, and as you know this kind of network has a latent layer. That would be great if we could project this latent layer in a lower dimension (3 in this case). This could be possible by adding this code:

```python
mu, logvar = model.encode(data.view(-1, 784))
z = model.reparameterize(mu, logvar)
summary_writer.add_embedding(z, global_step=epoch, tag='latent_layer')
```
encode and reparameterize are two parts of our network, and we show the output of reparameterize part based on our input data as the value of latent layer (z). The result would be like this: 

![projection](images/second-post/embedding.png)

#### 4- Loss function

Loss function is kind of a scalar value, and we can add this value to our summary with below snippet and the result is just like the following picture.

```python
summary_writer.add_scalar('training loss', train_loss, epoch)
```
We add epoch as the global_step which is neccessary for tensorboard to recognize the value based on the program's step. It is recommended to watch your loss function value after every epoch.

![loss function](images/second-post/loss.png)

#### Final note: 

There are many other types of summary which we cannot introduce here because of type of our project (unsupervised learning), and you can find them in [this address](https://pytorch.org/docs/stable/tensorboard.html).