
# S4

    ----------------------------------------------------PART- 1 -----------------------------------------------------



  Neural Network Architecture 
 
  ![image](https://user-images.githubusercontent.com/53977148/137504158-274818a2-750e-424e-b49b-1e9ca7273972.png)


  Forward Propagation Details 

    We can calculate an output from a neural network by propagating an input signal through each layer until the output layer .We call this forward- propagation.We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer.
  
Details Mathematical Calculation as below :

    input1=i1
    input2=i2
    actual_output=t1
    actual_output=t2
    h1=i1*w1+i2*w2
    h2=i1*w3+i2*w4
    act_h1=sigmoid(h1)
    act_h2=sigmoid(h2)
    o1= act_h1*w5 + act_h2*w6
    o2= act_h1*w7 + act_h2*w8
    act_o1=sigmoid(o1)
    act_o2=sigmoid(o2)
    E1=1/2*( t1 - act_o1)2
    E2=1/2*(t2 - act_o2)2
    E_total=E1+E2
    sigmoid(x)=(1/1+exp(-x))


  Back Propagation Details  
  
   The backpropagation algorithm is named for the way in which weights are trained.Error is calculated between the actual outputs labels and the outputs labels forward propagated from the network. These errors are then propagated backward through the network from the output layer to the hidden layer, updating weights as they go.

Details Mathematical Calculation as below :

    dsigmoid(x)/dx=sigmoid(x)*(1-sigmoid(x))
    dEtotal/dw5= d(E1+E2)/dw5= dE1/dact_o1*dact_o1/do1*do1/dw5	
    dE1/dact_o1=0.5*2*(t1 - act_o1)*-1 =act_o1 - t1		
    dact_o1/do1=dsigmoid(o1)/do1 =  act_o1 * (1-act_o1)		
    do1/dw5=d(act_h1*dw5)/dw5+d(act_h2*dw6)/dw5=act_h1*1+act_h2*0=act_h1	
    dE1/dact_o1=0.5*2*(t1 - act_o1)*-1 =act_o1 - t1
    dact_o1/do1=dsigmoid(o1)/do1 =  act_o1 * (1-act_o1)	
    do1/dw6=d(act_h1*dw5)/dw6+d(act_h2*dw6)/dw6=act_h1*0+act_h2*1=act_h2 
    dEtotal/dw7= d(E1+E2)/dw7= dE2/dact_o2*dact_o2/do2*do1/dw7						
    dE2/dact_o2=0.5*2*(t2 - act_o2)*-1 =act_o2 - t2						
    dact_o2/do2=dsigmoid(o2)/do2 =  act_o2 * (1-act_o2)						
    do2/dw7=d(act_h1*dw7)/dw7+d(act_h2*dw6)/dw7=act_h1*1+act_h2*0=act_h1						
					
    dEtotal/dw8= d(E1+E2)/dw8= dE2/dact_o2*dact_o2/do2*do1/dw8					
    dE2/dact_o2=0.5*2*(t2 - act_o2)*-1 =act_o2 - t2					
    dact_o2/do2=dsigmoid(o2)/do2 =  act_o2 * (1-act_o2)					
    do2/dw8=d(act_h1*dw7)/dw8+d(act_h2*dw6)/dw8=act_h1*0+act_h2*1=act_h2					

    do1/dact_h1=d(act_h1*w5+act_h2*w6)/dact_h1= w5							
    do1/dact_h2=d(act_h1*w5+act_h2*w6)/dact_h2= w6							
    dE1/dact_h1=   dE1/dact_o1*dact_o1/do1*do1/dact_h1  =  (act_o1 - t1)*act_o1*(1-act_o1) *w5							
    dE1/dact_h2=   dE1/dact_o1*dact_o1/do1*do1/dact_h2  =  (act_o1 - t1)*act_o1*(1-act_o1) *w6		
    do2/dact_h1=d(act_h1*w7+act_h2*w8)/dact_h1= w7							
    do2/dact_h2=d(act_h1*w7+act_h2*w8)/dact_h2= w8							
    dE2/dact_h1=   dE2/dact_o2*dact_o2/do2*do2/dact_h1  =  (act_o2 - t2)*act_o2*(1-act_o2) *w7							
    dE2/dact_h2=   dE2/dact_o2*dact_o2/do2*do1/dact_h2  =  (act_o2 - t2)*act_o2*(1-act_o2) *w8					
    dEtotal/dact_h1= d(E1+E2)/dact_h1= dE1/dact_h1+ dE2/dact_h1=     (act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7										
    dEtotal/dact_h2= d(E1+E2)/dact_h2= dE1/dact_h2+ dE2/dact_h2=     (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8										
	dact_h1/dh1= act_h1*(1-act_h1)
    dact_h2/dh2= act_h2*(1-act_h2)   
    dh1/dw1=i1	
    dh1/dw2=i2		
    dh2/dw3=i1		
    dh2/dw4=i2	

    dEtotal/dw1 = ((act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7) * act_h1*(1-act_h1)*i1						
    dEtotal/dw2 = ((act_o1 - t1)*act_o1*(1-act_o1) *w5 + (act_o2 - t2)*act_o2*(1-act_o2) *w7) * act_h1*(1-act_h1)*i2						
    dEtotal/dw3 = (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8) *act_h2*(1-act_h2) * i1						
    dEtotal/dw4 = (act_o1 - t1)*act_o1*(1-act_o1) *w6 + (act_o2 - t2)*act_o2*(1-act_o2) *w8) *act_h2*(1-act_h2) * i2						
    dEtotal/dw5 = (act_o1 - t1)*act_o1*(1-act_o1)*act_h1 	
    dEtotal/dw6 = (act_o1 - t1)*act_o1*(1-act_o1)*act_h2 
    dEtotal/dw7 = (act_o2 - t2)*act_o2*(1-act_o2)*act_h1 						
    dEtotal/dw8 = (act_o2 - t2)*act_o2*(1-act_o2)*act_h2
    
    w1 = old_w1 - learning_rate * dEtotal/dw1
    w2 = old_w2 - learning_rate * dEtotal/dw2
    w3 = old_w3 - learning_rate * dEtotal/dw3
    w4 = old_w4 - learning_rate * dEtotal/dw4
    w5 = old_w5 - learning_rate * dEtotal/dw5
    w6 = old_w6 - learning_rate * dEtotal/dw6
    w7 = old_w7 - learning_rate * dEtotal/dw7
    w8 = old_w8 - learning_rate * dEtotal/dw8

Image from Excel 

![Capture](https://user-images.githubusercontent.com/53977148/137503326-f124687a-bdb6-4c0f-bd37-c37c1c743d61.PNG)

![Capture1](https://user-images.githubusercontent.com/53977148/137502273-740bb820-a4b2-41fb-a3d1-638d3b3102e6.PNG)

![Capture3](https://user-images.githubusercontent.com/53977148/137502016-82f599d9-7fb5-4ebf-8774-329228560a62.PNG)

![Capture4](https://user-images.githubusercontent.com/53977148/137502046-7afa8b28-4b3e-4db7-83d2-b8b8b702d35c.PNG)

![Capture5](https://user-images.githubusercontent.com/53977148/137502060-660f36bf-c614-4ac2-8093-bafa43dd6135.PNG)

![Capture6](https://user-images.githubusercontent.com/53977148/137502115-f542bae8-6322-42b6-828e-c47006a88cc9.PNG)

![Capture7](https://user-images.githubusercontent.com/53977148/137502163-a6e5d92d-ecea-4e3e-8e33-3638a46a1417.PNG)

Error graph with Learning Rate 0.1 

![lr_0 1](https://user-images.githubusercontent.com/53977148/137502861-f3c1483a-316e-487a-a65f-6b891c50348c.PNG)

Error graph with Learning Rate 0.2

![lr_0 2](https://user-images.githubusercontent.com/53977148/137502866-34cc2084-9784-4020-bec9-cb1139c081df.PNG)

Error graph with Learning Rate 0.5 

![lr_0 5](https://user-images.githubusercontent.com/53977148/137502873-1dc2ac1b-d102-4e3e-92a6-c8e841579d94.PNG)

Error graph with Learning Rate 0.8

![lr_0 8](https://user-images.githubusercontent.com/53977148/137503028-7148e8ff-d5ad-4fb5-ac37-5fb8a002f30d.PNG)

Error graph with Learning Rate 1 

![lr_1](https://user-images.githubusercontent.com/53977148/137503040-f8bd63da-c4ef-462e-a30e-e93db96a68fe.PNG)

Error graph with Learning Rate 2 

![lr_2](https://user-images.githubusercontent.com/53977148/137503045-0d059ed6-a137-4c3a-a692-958ad6edd180.PNG)




----------------------------------------------------PART- 2 -----------------------------------------------------

Data Overview


MNIST ("Modified National Institute of Standards and Technology") dataset of computer vision. The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.This project implements a beginner classification task on MNIST dataset with a Convolutional Neural Network(CNN) model.

![image](https://user-images.githubusercontent.com/70502759/137764343-c1134fa1-94d2-40b0-bf21-dcd78b3ed4e1.png)
  
  This project will automatically dowload and process the MNIST dataset
  
  Design the model architecture for MNIST with following constraint :
    
    99.4% validation accuracy
    Less than 20k Parameters
    Less than 20 Epochs
    Have used BN, Dropout, a Fully connected layer, have used GAP. 
 
 Model Architecture 1 : With CNN and Linear NN at the last layer  

         
      
Requirement already satisfied: torchsummary in /usr/local/lib/python3.7/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
       BatchNorm2d-2           [-1, 32, 28, 28]              64
         MaxPool2d-3           [-1, 32, 14, 14]               0
            Conv2d-4           [-1, 32, 14, 14]           9,248
       BatchNorm2d-5           [-1, 32, 14, 14]              64
         MaxPool2d-6             [-1, 32, 7, 7]               0
            Conv2d-7             [-1, 32, 5, 5]           9,248
       BatchNorm2d-8             [-1, 32, 5, 5]              64
           Dropout-9             [-1, 32, 5, 5]               0
        AvgPool2d-10             [-1, 32, 1, 1]               0
           Linear-11                   [-1, 16]             528
      BatchNorm1d-12                   [-1, 16]              32
          Dropout-13                   [-1, 16]               0
           Linear-14                   [-1, 10]             170
    ================================================================

    Total params: 19,738
    Trainable params: 19,738
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.56
    Params size (MB): 0.08
    Estimated Total Size (MB): 0.64

      Result 

      Highest Accuracy with above architecture is around 99.00 to 99.11  


      
## Tech Stack

Client: Python, Pytorch

  