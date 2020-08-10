---
layout: post
title: Using Tensorboard in PyTorch
---

Watching closely parameters in neural networks is very important during the training process. Tensorboard is a well-known tool for this job, but some people think this tool is designed jsut for Tensorflow framework. However, I will show you on this post that Tensorboard can be used in PyTorch as well.

In this tutorial I used [an official example](https://github.com/pytorch/examples/tree/master/vae) of PyTorch about VAE network, and also final code of this project is available [here](https://github.com/mrhajbabaei/pytorch-tensorboard) (just be aware that I changed this example to be match with our expectations). At the first step we should import **SummaryWriter**; which plays the main role in creating a summay, and in the next steps I will show you how we can create different types of visualization in our program by this. At the next step, you should create an instance of **SummaryWriter**, so in top of your program in import section put this code:   
`from torch.utils.tensorboard import SummaryWriter`    
, and create an instance of it in your main function, or wherever you want to use it:    
`writer = SummaryWriter('your_log_directory/name_of_your_summary_experiment')`

**Note:** you should have your log directory already; consider something like this: "log" as your main directory for all kinds of summarization and experiment.    
**Note:** Just make sure log dire is excluded from your git repository (because of preventing upload unwanted big log files to your repository).

Generally, you can add different kinds of summary for your project based on type of your project and your needs. In this project I consider these summary types:
#### 1- Image grid   
As I mentioned before, this project is a VAE example and input for this project is [MNIST](http://yann.lecun.com/exdb/mnist/) database which cosist of 60,000, and 10,000 images with 28x28 size for training and testing respectively. To doing so, you should make a grid by your batch data and add this image grid to your summary writer:   
```python
image_grid = make_grid(your_data_batch)  
summary_writer.add_image('mnist_images', image_grid)
```
Please consider that your data batch could be each of batch data of your traning data, and the result is look like this:

![image_grid](https://user-images.githubusercontent.com/25500417/89778472-d65cd300-db22-11ea-8933-3b4825088f38.png)


#### 2- Graph  
You can have a full graph of entire your network. Just by adding this snippet you can add your graph to your summary:
```python
summary_writer.add_graph(model, your_data_batch)
```
Like above, you can use each of your data batch you would like, and the output would be like this:

![graph](https://user-images.githubusercontent.com/25500417/89778404-b3322380-db22-11ea-91fa-6bbcedd301cd.png)

You can click on each part to enlarge it:

![graph_zoom](https://user-images.githubusercontent.com/25500417/89778454-cc3ad480-db22-11ea-9acf-a774b11e8689.png)


#### 3- Embedding

We use VAE in this project, and as you know this kind of network has a latent layer. That would be great if we could project this latent layer in a lower dimension (3 in this case). This could be possible by adding this code:

```python
mu, logvar = model.encode(data.view(-1, 784))
z = model.reparameterize(mu, logvar)
summary_writer.add_embedding(z, global_step=epoch, tag='latent_layer')
```
encode and reparameterize are two parts of our network, and we show the output of reparameterize part based on our input data as the value of latent layer (z). The result would be like this: 

![embedding](https://user-images.githubusercontent.com/25500417/89778344-9564be80-db22-11ea-8fd7-f7b8a38468d9.png)
)

#### 4- Loss function

Loss function is kind of scalar value, and we can add this value to our summary with below snippet code, and the result is just like the following picture.

```python
summary_writer.add_scalar('training loss', train_loss, epoch)
```
Note that we add epoch as the global_step which is neccessary for tensorboard to synchronize the value based on the program's iterations. It is recommended to watch your loss function value after every epoch.

![loss](https://user-images.githubusercontent.com/25500417/89778514-e674b280-db22-11ea-8cdb-a63c6740d992.png)


#### Final note

There are some other types of summary which we cannot introduce here because of type of our project (unsupervised learning), and you can find them in [this address](https://pytorch.org/docs/stable/tensorboard.html).

#### References
1- [PyTorch Tensorboard tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)   
2- [PyTorch VAE official example](https://github.com/pytorch/examples/tree/master/vae)