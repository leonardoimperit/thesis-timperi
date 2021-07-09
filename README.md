<h1 align="center"> Exploiting Generative Adversarial Networks for credit card fraud detection </h1>


<h2> Introduction </h2>
The purpose of this project is to implement a Generative Adversarial Network as an oversampling method for the imbalanced problem of credit card fraud detection. We will propose it as an alternative to the very popular SMOTE oversampling technique. For this purpose we will employ 2 datasets: a bigger one composed by simulated transactions (CHAPTER 5) through a credit transaction dataset simulator, and a smaller one contaning only 2 days of  real transactions (CHAPTER 6). For the first one, the simulator has been developed by the ULB machine learning group (Bruxelles), the latter has been downloaded from Kaggle. It contains 284,807 transactions, 492 of which have been classified as frauds.  Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.

<h2> Our Goals: </h2>
<ul>
<li> Understand the  class and features distribution of our Datasets </li>
<li> Implement different oversampling and undersampling techniques to deal with our imbalanced dataset</li>
<li> Determine the Classifiers we are going to use based on specific performance measures other than from Accuracy </li>
<li> Analyze the economic impact of our algorithm on an average company in the market</li>
</ul>


<h2> Index: </h2>

**CHAPTER 5: <b>APPLICATION ON A SIMULATED DATASET**  </b><br></b>

**5.1) Gather sense of our data** <br>
5.1.1) [Functioning of the transaction simulator](#da) <br>
5.1.2) [Data description and variable transformation](#da) <br>
5.1.3) [Base-line classification without oversampling](#da) <br> <br> 

**5.2) GAN oversampling** <br>
5.2.1) [Creation of a Generative Adversarial Network](#distributing)<br>
5.2.2) [Classification models](#da)<br>
5.2.3) [Performance](#da)<br><br>

**5.3) SMOTE oversampling**</b><br>
5.3.1) [SMOTE implementation](#da)<br>
5.3.2) [Classification and performance](#da)<br><br>



**CHAPTER 6: <b>APPLICATION ON A REAL DATASET**  </b><br></b>

**6.1) Gather Sense of our data** <br>
6.1.1) [Data description and variable transformation](#da) <br><br>
**6.2) GAN OVERSAMPLING**</b><br>
6.2.1) [ Tuning of the Generative Adversarial Networkk](#da)<br>
6.2.2) [Model performance ](#da)<br>
6.2.3) [Performance](#da)<br><br>

6.3 <b>SMOTE oversampling</b><br>
6.3.1) [Classification and performance](#da)<br>



<h2> References: </h2>
<ul> 
<li>Hands on Machine Learning with Scikit-Learn & TensorFlow by Aurélien Géron (O'Reilly). CopyRight 2017 Aurélien Géron  </li>
<li><a src="https://www.youtube.com/watch?v=DQC_YE3I5ig&t=794s" > Machine Learning - Over-& Undersampling - Python/ Scikit/ Scikit-Imblearn </a>by Coding-Maniac</li>
<li><a src="https://www.kaggle.com/lane203j/auprc-5-fold-c-v-and-resampling-methods"> auprc, 5-fold c-v, and resampling methods
</a> by Jeremy Lane (Kaggle Notebook) </li>
</ul>
