---
layout: post
title: PCA and how we can use it for Covid-19
---

#### Introduction
After reading the Unsupervised Learning chapter in [An Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/) book (Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani), I decided to write a concise article about the PCA and how to apply it on a Covid-19 dataset.
 
#### PCA in a nutshell
Unsupervised Learning has been becoming popular among the scientist in various aspects including medical scientist to predict cancer or ads companies to analyze the effect of their work on people.
In this chapter of the book, we can see two methods that are vastly used for Unsupervised Learning task, PCA and K-Means clustering. PCA mainly use to show data based on its Principal Components. As a case in point, you can show your 2D data just based on your first principal component if your features have a strong relationship; otherwise, you also should use the second component to have a more accurate prediction. In fact, we can have the same number of principal components just equal to the number of features. But, the idea behind PCA is to use lower features (principal components) than the total number of features of data. So, it is necessary to have correlated data to use fewer principal components, unless you cannot have a minimized mean square error.
The idea is to substitute our n observation with dimension p by a lower dimension representation. Provided that we have: <img src="https://user-images.githubusercontent.com/25500417/97175107-3836eb80-17a8-11eb-8285-3a27bc2713d4.png" width="80"> the first principal component would be a normalized linear combination of the features with maximal variance from each other:
<img src="https://user-images.githubusercontent.com/25500417/97175416-b85d5100-17a8-11eb-80c1-19ea9a698d00.png" width="250">
Every element (score values) has largest sample variance subject to this constraint:
<img src="https://user-images.githubusercontent.com/25500417/97175677-1f7b0580-17a9-11eb-8ab8-139ac94d3be5.png" width="80">

There is a geometry that says data mostly varied by the first principal component direction. The second principal component has an additional constraint by which <img src="https://user-images.githubusercontent.com/25500417/97175818-53562b00-17a9-11eb-9407-1792a5f4c18d.png" width="20"> should be uncorrelated with <img src="https://user-images.githubusercontent.com/25500417/97176049-a03a0180-17a9-11eb-8536-940e4aadeae8.png" width="20"> (direction of <img src="https://user-images.githubusercontent.com/25500417/97176116-b9db4900-17a9-11eb-9ec2-5d5331178da9.png" width="20"> is orthogonal to the <img src="https://user-images.githubusercontent.com/25500417/97176205-d8d9db00-17a9-11eb-8f03-dcb734cbfd0a.png" width="20">'s direction). I should also mention that <img src="https://user-images.githubusercontent.com/25500417/97176285-f6a74000-17a9-11eb-9330-b41c1ce0bcc3.png" width="20"> vectors also call loading vectors, and <img src="https://user-images.githubusercontent.com/25500417/97176365-15a5d200-17aa-11eb-985b-c659ba380776.png" width="20"> vectors call score vectors.


#### Features
PCA has some features including uniqueness through which you can find unique principal components by different programming tools. Besides, maybe they have different signs, but it does not matter because the only important thing is the principal components' direction. 
Another handy feature is scaling. In fact, you should scale your variable before applying PCA when you have a dominant variable with very high variance unless your first principal component would be in the direction of that variable, and your result would be scattered in the direction of it.


#### PCA in Python
In the previous section, we briefly learn what PCA is, and now we want to try PCA on real data. We use [Covid-19 dataset](https://github.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning) in which there are various collections of Covid-19 related datasets. I consider using **_full_grouped.csv_** file through this tutorial. In this file, there are 10 columns regarding Date (last update of the database), Country/Region, Confirmed, Deaths, Recovered, Active, New cases, New deaths, New recovered, and WHO Region. 
![Pic-01](https://user-images.githubusercontent.com/25500417/97176806-bbf1d780-17aa-11eb-8c33-044d7bed6cf2.jpg)
First, we want to apply PCA for continents. To do so, we need to remove unnecessary columns such as **Date** and **Country/Region**, and by putting this code, we can do that:
```python
df = df['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered', 'WHO Region']
```
Second, we should aggregate all data related to every specific continent in a single row. To do so, we should use **_group by_** and **_sum()_** command:
```python
df = df.groupby(['WHO Region']).sum().reset_index()
```
The result would be like this:
![Pic-02](https://user-images.githubusercontent.com/25500417/97176934-e774c200-17aa-11eb-848e-673bab4b41ce.jpg)

Now, if we apply PCA on this data, the output would be like the below image (the result of applying the PCA on our dataset at the top, and loading vectors of the PCA algorithm at the bottom):
<img src="https://user-images.githubusercontent.com/25500417/97180580-7edc1400-17af-11eb-9f4d-c28d5880b7f4.jpg" height="400">

You can find the Python code of this post at [this github address](https://github.com/mrhajbabaei/unsupervised-learning-covid-19).

#### Insights
By looking at the above charts, we can infer some important information. First, it seems Western Pacific, South-East Asia, Africa, and Eastern Mediterranean were approximately in the same group in terms of struggling against Covid-19. It makes sense because by looking at Covid-19 statistics at Google or the following news, you can see that the new death rate in countries in these territories is decreasing. By contrast, while Europ in the top chart is at the top-right side by which we can find that the death rate is relatively high and also recovered cases increased, Americas is in the bottom-right side of the chart which conveys information about that the rate of active and new cases are increased.

#### Conclusion
In conclusion, unsupervised learning methods, especially the PCA technique, are prevalent and useful for both visualization and dimension reduction. I wanted to show you how to use the PCA through this post using a real Covid-19 dataset (I hope this disease ends up very soon, and please wear a mask!).