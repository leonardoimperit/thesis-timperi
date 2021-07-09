<h1 align="center"> Exploiting Generative Adversarial Networks for credit card fraud detection </h1>


<h2> Introduction </h2>
The purpose of this project is to implement a Generative Adversarial Network as an oversampling method for the imbalanced problem of credit card fraud detection. We will propose it as an alternative to the very popular SMOTE oversampling technique and Random Undersampling. The dataset "credit card fraud detection" has been downloaded from Kaggle. It contains 284,807 transactions, 492 of which have been classified as frauds.  Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.

<h2> Our Goals: </h2>
<ul>
<li> Understand the  class and features distribution of our Dataset </li>
<li> Implement different oversampling and undersampling techniques to deal with our imbalanced dataset</li>
<li> Determine the Classifiers we are going to use based on specific performance measures other than from Accuracy)  </li>
<li> Analyze the economic impact of our algorithm on an average company in the market</li>
</ul>


<h2> Index: </h2>

I. <b>Data Analysis </b><br>
1.1) [Gather Sense of our data](#da) <br><br>

II. <b>Data preparation</b><br>
2.1) [Data cleaning](#distributing)<br>
2.2) [Scaling and Distributing](#distributing)<br>
2.3) [Train and Test set creation](#splitting)<br><br>

III. <b>Oversampling with GAN</b><br>
3.1) [Preparing the dataset for GAN learning](#correlating)<br>
3.2) [GAN definition](#anomaly)<br>
3.3) [Generating new data using GAN](#anomaly)<br>
3.4) [Classification without cross validation](#anomaly)<br>
3.4) [Cross Validation with GAN](#anomaly)<br> 
3.5) [Performance and testing](#logistic)<br><br> 


IV. <b>SMOTE Oversampling</b><br>
4.1) [Classifiers and Cross Validation](#classifiers)<br>
4.2) [Performance and Testing](#logistic)<br><br> 

V. <b>Model comparison </b><br>
5.1) [Confusion matrices](#testing_logistic)<br>
5.2) [Economic Analysis](#neural_networks)


<h2> References: </h2>
<ul> 
<li>Hands on Machine Learning with Scikit-Learn & TensorFlow by Aurélien Géron (O'Reilly). CopyRight 2017 Aurélien Géron  </li>
<li><a src="https://www.youtube.com/watch?v=DQC_YE3I5ig&t=794s" > Machine Learning - Over-& Undersampling - Python/ Scikit/ Scikit-Imblearn </a>by Coding-Maniac</li>
<li><a src="https://www.kaggle.com/lane203j/auprc-5-fold-c-v-and-resampling-methods"> auprc, 5-fold c-v, and resampling methods
</a> by Jeremy Lane (Kaggle Notebook) </li>
</ul>
