# FederatedLearning
Import all relevant packages
Don’t worry, I will provide details for each of the imported modules at the point of instantiating their respective objects.
Reading and preprocessing MNIST data set
I’m using the jpeg version of MNIST data set from here. It consists of 42000 digit images with each class kept in separate folder. I will load the data into memory using this code snippet and keep 10% of the data for testing the trained global model later on.
On line 9, each image will be read from disk as grey scale and then flattened. The flattening step is import because we will be using a MLP network architecture later on. To obtain the class label of an image, we split its path string on line 11. Hope you noticed we also scaled the image to [0, 1] on line 13 to douse the impact of varying pixel brightness.
Creating train-test split
A couple of steps took place in this snippet. We applied the load function defined in the previous code block to obtain the list of images (now in numpy arrays) and label lists. After that, we used the LabelBinarizer object from sklearn to 1-hot-encode the labels. Going forward, rather than having the label for digit 1 as number 1, it will now have the form[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]. With this labelling style, we’ll be able to use the cross-entropy loss in Tensorflow as our model’s loss function. Alternatively, I could have left the labels as it was and use the sparse-categorical-entropy loss instead. Finally, I used the sklearn’s train_test_split object to split the data into a train/test with ratio 9:1.
Federated Members (clients) as Data Shards
In the real world implementation of FL, each federated member will have its own data coupled with it in isolation. Remember the aim of FL is to ship models to data and not the other way around. The shard creation step here only happens in experiments. I will share the training set into 10 shards, one per client. I wrote a function called create_clients to achieve this.
On line 13, I created a list of client names using the prefix (initials). On line 16–20, I zipped the data and label lists then randomised the resulting tuple list. Finally I created shards from the tuple list based on the desired number of clients (num_clients) on line 21. On line 26, a dictionary containing each client’s name as key and their data share as value was returned. Let’s now go ahead and apply this function to our training data set.
Processing and batching clients’ and test data
Next is to process each of the client’s data into tensorflow data set and batch them. To simplify this step and avoid repetition, I encapsulated the procedure into a small function named batch_data.
I trust you remember that each of the client data sets came out as a (data, label) tuple list from create_clients. On line 9 above, I split the tuple into separate data and label lists. I then made a shuffled and batched tensorflow dataset object off these lists.

While applying this function below, I will process the test set as well and keep it aside for later use.

Creating the Multi Layer Perceptron (MLP) model
One thing I didn't mention in the introduction section is that FL is mostly suited for parameterized learning — all types of neural networks. Machine learning techniques such as KNN or it likes that merely store training data while learning might not benefit from FL. I’m creating a 3-layer MLP to serve as the model for our classification task. I hope you still remember all those Keras modules we imported earlier, this is where they fit in.
To build a new model, the build method will be invoked. It requires the input data’s shape and the number of classes as arguments. With MNIST, the shape parameter will be 28*28*1 = 784,while the number of classes will be 10.

Now is the time to define an optimizer, loss function and metrics to compile our models with later on.

SGD is my default optimizer except when I have a reason not to use it. The loss function is categorical_crossentropy. And finally, the metric I will be using is accuracy. But something looks strange in the decay argument. What’s comms_round? It’s simply the number global epochs (aggregations) I will be running during training. So rather than decaying the learning rate with respect to the number of local epochs as you might be familiar with, here I want to decay with respect to the number of global aggregation. This is obviously an hyper parameter selection choice, but I found it to work pretty well while experimenting. I also found an academic report where this setting worked too [1].

Model Aggregation (Federated Averaging)
All I have done up to this point was pretty much standard as per deep learning pipeline. Of course with the exception of the data partitioning or client creation bit. I will now move on to Federated Averaging ( the vanilla algorithm for FL) which is the whole point of the this tutorial. The data I’m using is horizontally partitioned, so I will simply be doing component wise parameter averaging which will be weighed based on the proportion of data points contributed by each participating client. Here’s the federated averaging equation I’m using, it comes one of the pioneering works on federated learning [2].



Don’t let the complex mathematical notations in the equation fool you, this is a pretty straight forward computation. On the right hand side, we are estimating the weight parameters for each client based on the loss values recorded across every data point they trained with. On the left, we scaled each of those parameters and sum them all component-wise.

![image](https://user-images.githubusercontent.com/97829880/218524289-5048ef2f-b369-4881-9804-fcd1f0c456ae.png)

Below I have encapsulated this procedure into three simple functions.

(1) weight_scalling_factor calculates the proportion of a client’s local training data with the overall training data held by all clients. First we obtained the client’s batch size and used that to calculate its number of data points. We then obtained the overall global training data size on line 6. Finally we calculated the scaling factor as a fraction on line 9. This sure can’t be the approach in a real world application. The training data will be disjointed, therefore no single client can correctly estimate the quantity of the combined set. In that case, each client will be expected to indicate the number of data points they trained with while updating the server with new parameters after each local training step.

(2) scale_model_weights scales each of the local model’s weights based the value of their scaling factor calculated in (1)

(3) sum_scaled_weights sums all clients’ scaled weights together.

Federated Model Training

The training logic has two main loops, the outer loop is for the global iteration, the inner is for iterating through client’s local training. There’s an implicit third one though, it accounts for the local epochs and will be taken care of by the epochs argument in our model.fit method.

Starting out I built the global model with input shape of (784,) and number of classes as 10 — lines 2–3. I then stepped into the outer loop. First obtaining the initialised weights of the global model on line 9. Lines 15 and 16 shuffles the clients dictionary order to ensure randomness. From there, I started iterating through client training.

For each client, I created a new model object, compiled it and set it’s initialisation weights to the current parameters of the global model — lines 20–27. The local model (client) was then trained for one epoch. After training, the new weights were scaled and appended to the scaled_local_weight_list on line 35. That was it for local training.

Moving back into the outer loop on line 41, I summed up all the scaled local trained weights (of course by components) and updated the global model to this new aggregate. That ends a full global training epoch.

I ran 100 global training loops as stipulated by the comms_round and on line 48 tested the trained global model after each communication round our test data. Here is the snippet for the test logic:

Results
With 10 clients each running 1 local epoch on top of 100 global communication rounds, here is the truncated test result:

SGD Vs Federated Averaging
Yes, our FL model results are great, 96.5% test accuracy after 100 communication rounds. But how does it compare to a standard SGD model trained on the same data set? To find out, I’ll train a single 3-layer MLP model (rather 10 as we did in FL) on the combined training data. Remember the combined data was our training data prior to partitioning.

To ensure an equal playing ground, I will retain every hyper parameter used for the FL training except the batch size. Rather than using 32 , our SGD’s batch size will be 320. With this setting, we are sure that the SGD model would see exactly the same number of training samples per epoch as the global model did per communication round in FL.

There you have it, a 94.5% test accuracy for the SGD model after 100 epochs. Isn’t it surprising that the FL performed a little better than its SGD counterpart with this data set? I warn you not to get too excited about this though. These kind of results are not likely in real world scenario. Yeah! Real world federated data held by clients are mostly NON independent and identically distributed (IID).

For example, we could have replicated this scenario by constructing our client shards above such that each comprises of images from a single class — e.g client_1 having only images of digit 1, client_2 having only images of digit 2 and so on. This arrangement would have lead to a significant reduction in the performance of the FL model. I leave this as an exercise for the reader to try out. Meanwhile, here is the code you could use to shard any classification data in a non-IID manner.

Conclusion
Through this article, I introduced the concept of Federated Learning and took you through the tensorflow implementation of it basic form. I encourage you to check my recent article on LinkedIn here for broader introduction of this technology, particularly if you are not clear about its workings or want to learn more about how it could be applied. For researchers wanting to study this subject in more depth, there are lots of journals around FL on arxiv.org/cs , mostly pushing boundaries on its implementation and addressing its numerous challenges.

Reference
[1] Federated Learning with Non-IID Data, Yue Zhao et al, arXiv: 1806.00582v1, 2 Jun 2018

[2] Communication-Efficient Learning of Deep Networks from Decentralized Data, H. Brendan McMahan et al, arXiv:1602.05629v3 [cs.LG] 28 Feb 2017
