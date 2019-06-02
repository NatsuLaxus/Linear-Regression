In [8]:
#import the necessary libraries first
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
In [9]:
np.random.seed(101) 
tf.set_random_seed(101) 
# Genrating random linear data 
# There will be 50 data points ranging from 0 to 50 
x = np.linspace(0, 50, 50) 
# Adding noise to the random linear data 
x += np.random.uniform(-4, 4, 50)
'''
line_1:

Now generate the value of Y randomly from standard normal distribution. 
Make sure the shape of X and Y are same

'''
y = np.linspace(-2,2,50)
y += np.random.uniform(-2,2,50)


n = len(x) # Number of data points 
In [10]:
# Plot of Training Data 
plt.scatter(x, y) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title("Training Data") 
plt.show() 

In [11]:
X = tf.placeholder("float") 
Y = tf.placeholder("float") 
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b")
'''
line_2 & line_3 : 
Now , create two variables named learning_rate and training_epochs and set some value. 
First assign learning rate as 0.01 and training epochs as 1000
'''
learning_rate = 0.01
training_epochs = 200

'''
line_4:

declare the hypothesis line as 

y_pred= X*W + b

use tensorflow (tf) to add and mutiply 
'''
y_pred = tf.add(tf.multiply(X,W),b)
'''
line_5:

Declare the cost function as mean squared error of y_pred and Y as

         1                         2 
cost =  --- [ sum of ( y_pred - Y )   ]
         2n
         
Use tf.reduce_sum and tf.pow 

'''
cost = tf.reduce_sum(tf.pow((y_pred - Y),2) / (2 * n))


# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# Global Variables Initializer 
init = tf.global_variables_initializer() 
In [12]:
# Starting the Tensorflow Session 
with tf.Session() as sess: 
    sess.run(init) 
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
        # Feeding each data point into the optimizer using Feed Dictionary 
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b) 
Epoch 50 : cost = 0.61677545 W = 0.022213679 b = -0.49482715
Epoch 100 : cost = 0.6048826 W = 0.020790085 b = -0.42521897
Epoch 150 : cost = 0.5956166 W = 0.019518534 b = -0.3630452
Epoch 200 : cost = 0.588423 W = 0.018382818 b = -0.30751306
Epoch 250 : cost = 0.58286124 W = 0.017368414 b = -0.2579128
Epoch 300 : cost = 0.5785825 W = 0.016462374 b = -0.213611
Epoch 350 : cost = 0.57531047 W = 0.015653118 b = -0.17404154
Epoch 400 : cost = 0.5728264 W = 0.014930313 b = -0.13869937
Epoch 450 : cost = 0.5709573 W = 0.014284714 b = -0.10713214
Epoch 500 : cost = 0.5695671 W = 0.013708079 b = -0.07893695
Epoch 550 : cost = 0.5685478 W = 0.013193037 b = -0.053753447
Epoch 600 : cost = 0.5678151 W = 0.012733014 b = -0.03126006
Epoch 650 : cost = 0.5673023 W = 0.01232213 b = -0.011169439
Epoch 700 : cost = 0.5669574 W = 0.011955135 b = 0.0067751193
Epoch 750 : cost = 0.5667394 W = 0.011627343 b = 0.022802811
Epoch 800 : cost = 0.56661665 W = 0.011334568 b = 0.037118427
Epoch 850 : cost = 0.56656444 W = 0.011073064 b = 0.049904883
Epoch 900 : cost = 0.56656355 W = 0.010839496 b = 0.061325517
Epoch 950 : cost = 0.56659925 W = 0.010630878 b = 0.07152609
Epoch 1000 : cost = 0.5666603 W = 0.010444543 b = 0.080637135
In [13]:
# Calculating the predictions 
predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 
Training cost = 0.5666603 Weight = 0.010444543 bias = 0.080637135 

In [14]:
# Plotting the Results 
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show() 

In [ ]: