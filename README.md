# NLP---disasters
Classifying if a tweet is a real natural disaster or not

In NLP we want to remove as many useless words as possible (in NLP jargon, known as stop-words, e.g the, a , but, ). Hence, text cleaning is very important step. Remember, this is not feature engineering. Neither is normalization, hyperparameter-tuning, etc..
corpus from NLTK library is perfect for removings stop words
We also might want to stem our words, e.g loved -> love should be the same. 
So, our quest is to classify a real natural disaster tweet. For example, our algorithm should be able to identify "the food at King Burgers is fire!" as a false positive,
and "King's Burgers is on fire" as a positive identification of a disaster. 


Examples of cleaning text;
1. removing numbers from words
2. removing stop words
3. removing special characters like *&^@#$@



Classification using Deep Learning

Activation functions
Theshold, 0 if false, 1 if true also for binary variable
Sigmoid, useful for getting probability , more like logistic regression <0.5 and .0.5 also usefull for binary variable, softmax for multiple
Activation functions are found in hidden layers, use relu for a connected one

How the whole process works?
Input value -> hidden layer, activation function -> compares to actual value, updates the weights to correct the error, minimize cost function
 
In artificial neural networks, hidden layers are required if and only if the data must be separated non-linearly.
#number of hidden layers depnds on case by case:
#0 only capable of representing linear separable decision
#1 approximate any function that contains a continous mapping from the one finite spae to another
#2 can present abitrary decision boundaries to accuracy

2a) ITERATIVE: Gradient Descent (optimization algorithm) <- tweak parameters to minimize cost function (MSE)
MEasures local gradient of error function related to vector  and goes till it reaches minimum
Learning rate if too small= many iterations to converge
FOr Gradient Descent, make sure you have a similar sclare StandardScaler().

2b) Stochastic GD SGDRegressor()
Stochastic means random
One instance needs to be in memory at each iteration.
Instead of decreasing to reach min, it bounces up and down and only decreases on average
Stochastic is faster, for ANN optimizer = 'adam'


How do you know you ANN is overfitting? In other classic machine learning algorithms, we have done some cross validation. But how do we do that for ANN? Just build a Kerasclassifier and make a model. Then you can do grid search and the likes!



