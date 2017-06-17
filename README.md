# Gradient Descent Coding Challenge
Code written by: Woratana Perth Ngarmtrakulchol

This is the code for "Intro - The Math of Intelligence" by Siraj Raval on [Youtube](https://youtu.be/xRJCOz3AfYY)

## Overview
This is the code for [this](https://youtu.be/xRJCOz3AfYY) video on Youtube by Siraj Raval.

The data set used here is `House Sales in King County, USA` from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction).

Note that instead of having `m` and `b` in `y=mx+b` as seperate variables, I added both variables in `weight`.

This way, we can predict the price by matrix multiplication. It served 2 purposes: shortening code (no need for looping through each data point) and faster computation (computer can work faster with matrix)

From the derived forms of `m` (theta1 in the image below) and `b` (theta0), we achieved them by using matrix multiplication with `intercept` column = 1.
![bgd_eqn](http://file.designil.com/es8q2+)
Batch Gradient Descent Equation from: http://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php

## More about Gradient Descent
Here are some helpful links:

#### Gradient descent visualization
https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif

#### Sum of squared distances formula (to calculate our error)
https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png

#### Partial derivative with respect to b and m (to perform gradient descent)
https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

## Dependencies

* numpy - For matrix multiplication
* pandas - Maybe unnecessary, but made the code a lot easier to read
* sklearns - For scaling input and output

## Usage

Just run ``python gd.py`` to see the results:

   ```
Starting gradient descent at weight = [ 0.02193762  0.11478817], error = 0.8513764945994572
Running...
After 100 iterations weight = [ 0.53208446  0.62493501], error = 0.5096657356181261
   ```

## Credits

Thank you Matt Nedrich and Siraj Raval for the sample code. I did rewrite all the functions, but following the same structure.
