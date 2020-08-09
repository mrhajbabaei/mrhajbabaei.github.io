---
layout: post
title: Using Tensorboard in PyTorch
---

Watching closely every parameter and vital measurements on-line is very important in training process of deep learning models. Tensorboard recognise as a great tool for this job, but some people use think this tool is designed jsut for Tensorflow framework. However, I will show you on this post that Tensorboard can be used for PyTorch as visualization tool as well.

In this tutorial I used [an official example](https://github.com/pytorch/examples/tree/master/vae) of PyTorch about VAE network, and also final code is available [here](https://github.com/mrhajbabaei/pytorch-tensorboard). At the first step we should import **SummaryWriter**; this plays the main role in creating a summay, and I I will show you how we can use this to create different types of visualization in our program. After that, you should create an instance of **SummaryWriter**, so in top of your program in import section put this code:
`from torch.utils.tensorboard import SummaryWriter`
, and in your main function, or wherever you want to use that create an instance of it:
`writer = SummaryWriter('your_log_directory/name_of_your_summary_experiment')`
**Note:** you should create your log dir already; consider something like this: "log_dir" as your main directory for all kinds of summarization and experiment.


