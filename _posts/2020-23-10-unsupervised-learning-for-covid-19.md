---
layout: post
title: PCA and how to use it on a Covid-19 dataset
---

#### Introduction
After reading the Unsupervised Learning chapter in "An Introduction to Statistical Learning" book (Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani), I decided to write a concise article about that by using apply it on a Covid-19 dataset.
 
#### PCA in a nutshell
Unsupervised Learning has been becoming popular among the scientist in various aspects including medical scientist to predict cancer or ads companies to analyze the effect of their work on people.
In this chapter of the book, we can see two methods that are vastly used for Unsupervised Learning task, PCA and K-Means clustering. PCA mainly use to show data based on its Principal Components. As a case in point, you can show your 2D data just based on your first principal component if your features have a strong relationship; otherwise, you also should use the second component to have a more accurate prediction. In fact, we can have the same number of principal components just equal to the number of features. But, the idea behind PCA is to use lower features (principal components) than the total number of features of data. So, it is necessary to have correlated data to use fewer principal components, unless you cannot have a minimized mean square error.
The idea is to substitute our n observation with dimension p by a lower dimension representation. Provided that we have $X_{1}$, $X_{2}$, ..., $X_{p}$ the first principal component would be a normalized linear combination of the features with maximal variance from each other
$Z_{1}$ = $\Phi_{11}$$X_{1}$ + $\Phi_{21}$$X_{2}$ + ... + $\Phi_{p1}$$X_{p}$ 
that has largest sample variance subject to this constraint
$\sum_{j=1}^n \Phi_{j1}^2 = 1$
There is a geometry that says data mostly varied by the first principal component direction. The second principal component has an additional constraint by which $Z_{2}$ should be uncorrelated with $Z_{1}$ (direction of $\Phi_{2}$ is orthogonal to the $\Phi_{1}$'s direction). I should also mention that $\Phi_{n}$ vectors also call loading vectors, and $Z_{n}$ vectors call score vectors.


#### Features
PCA has some features including uniqueness through which you can find unique principal components by different programming tools. Besides, maybe they have different signs, but it does not matter because the only important thing is the principal components' direction. 
Another handy feature is scaling. In fact, you should scale your variable before applying PCA when you have a dominant variable with very high variance unless your first principal component would be in the direction of that variable, and your result would be scattered in the direction of it.


#### PCA in Python
In the previous section, we briefly learn what PCA is, and now we want to try PCA on real data. We use [Covid-19 dataset]([www.github.com](https://github.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning)) in which there are various collections of Covid-19 related datasets. I consider using **_full_grouped.csv_** file through this tutorial. In this file, there are 10 columns regarding Date (last update of the database), Country/Region, Confirmed, Deaths, Recovered, Active, New cases, New deaths, New recovered, and WHO Region. 
![Pic-01](https://user-images.githubusercontent.com/25500417/97104419-fedb7e80-16c8-11eb-9cde-c84e146b55fc.jpg)
First, we want to apply PCA for continents. To do so, we need to remove unnecessary columns such as **Date** and **Country/Region**, and by putting this code, we can do that:
```python
df = df['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths', 'New recovered', 'WHO Region']
```
Second, we should aggregate all data related to every specific continent in a single row. To do so, we should use **_group by_** and **_sum()_** command:
```python
df = df.groupby(['WHO Region']).sum().reset_index()
```
The result would be like this:
![Pic-02](https://user-images.githubusercontent.com/25500417/97104574-2f6fe800-16ca-11eb-829d-0b5175a19994.jpg)

Now, if we apply PCA on this data, the output would be like the below image:
![Pic-03](https://user-images.githubusercontent.com/25500417/97144628-0870ef00-177a-11eb-9017-99b781746022.jpg)



#### Insights
By looking at the above charts, we can infer some important information. First, it seems Western Pacific, South-East Asia, Africa, and Eastern Mediterranean were approximately in the same group in terms of struggling against Covid-19. It makes sense because by looking at Covid-19 statistics at Google or the following news, you can see that the new death rate in countries in these territories is decreasing. By contrast, while Europ in the top chart is at the top-right side by which we can find that the death rate is relatively high and also recovered cases increased, Americas is in the bottom-right side of the chart which conveys information about that the rate of active and new cases are increased.

#### Conclusion
In conclusion, unsupervised learning methods, especially the PCA technique, are prevalent and useful for both visualization and dimension reduction. I wanted to show you how to use the PCA through this post using a real Covid-19 dataset (I hope this disease ends up very soon and wear a mask!).