#Importing necessary libraries
import numpy as np
np.random.seed()

#Generating random input and output data

x = np.random.rand(3,4)
y = np.random.randint(2,size=(3, 1))
print(y)

#sigmoid function

def sigmoid (x):
    return 1/(1+np.exp(-x))

#Derivative of a sigmoid function

def sigmoid_der(x):
    return sigmoid(x)*(1 - sigmoid(x))  
    


m = x.shape[0] #Number of training examples

l = 1 #Number of hidden layers

features = x.shape[1] #Number of features

hidden_nodes = 5  #Number of nodes in hidden layer

output_labels = 1 #Number of neurons in output layer

#Initializing weights and biases 

wh = np.random.rand(features,hidden_nodes)  
bh = np.random.randn(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_labels)  
bo = np.random.randn(output_labels)  
lr = 0.05

for epoch in range(10000):  

    #Forward Propogation

    zh = np.dot(x, wh) + bh
    ah = sigmoid(zh)

    
    zo = np.dot(ah, wo) + bo
    y_pred = sigmoid(zo)

    # Displaying the result after every 200 epochs
    
    if (epoch+1) % 200 == 0:
        # Calculating the cost a every epoch 
        cost = (1/m) * np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
        print("Epoch", (epoch + 1), ": cost =", cost) 
    #Back Propogation

    dcost_dzo = y_pred - y

    dcost_wo = (1/m)*np.dot(ah.T, dcost_dzo)

    dcost_bo = (1/m)*np.sum(dcost_dzo)

    dcost_dah = np.dot(dcost_dzo , wo.T)
    
    dah_dzh = sigmoid_der(zh)

    dcost_dzh = dah_dzh * dcost_dah

    dcost_wh = (1/m)*np.dot(x.T, dcost_dzh)

    dcost_bh = (1/m)*np.sum(dcost_dzh)

    # Updating Weights

    wh -= lr * dcost_wh
    bh -= lr * dcost_bh

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo

    
        

# Calculating the predictions 
zo = np.dot(ah, wo) + bo
y_pred = sigmoid(zo)

print("Output = " ,y_pred)
