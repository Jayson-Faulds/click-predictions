# click-predictions
In this project, I (and my team) attempt to predict whether an ad will get clicked by the user.

We utilized Lasso Regression, Random Forest, and Catboost methods to do these predictions. Ultimately, Random Forest performed the worst,
Lasso performed quite well, and Catboost did the best overall. While accuracy was not obtained, we were able to achieve a high precision
of 91.7% with a recall of 57%. We are especially proud of the recall statistic, which implies that whenever there was a click in the test
data, our model correctly predicted it 57% of the time. To do this, we had to balance the training data so that we had roughly similar
amounts of "clicks" and "no-clicks". This most likely had a negative effect on overall accuracy, but we think it is more important to 
detect clicks when they happen in this scenario.

The AWS R script contains all of our data cleaning, as well as the random forest model. The original data is from a kaggle competition 
from a few years ago: https://www.kaggle.com/c/avazu-ctr-prediction  It is too big to upload here onto GitHub. A recommendation, 
should you wish to run our code: the training data is quite massive. We only trained the model on 1 million rows, while the training data
from Kaggle has at least 12 million. To do this, we ran a bash script to randomly sample 1 million rows from the full dataset, and we used
that to train our model.

Bash: shuf Train.csv > TrainPerm.csv
      head -n 1000000 TrainPerm.csv > TrainPerm-1MM.csv
      
The lasso R script contains the code for the lasso regression, and the catboost notebook contains the code for our catboost model. 

Thank you for reading!
