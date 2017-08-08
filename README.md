# Expert-Gate
We introduce a model of lifelong learning, based on a Network of Experts. New tasks / experts are learned and added to the model sequentially, building on what was learned before. To ensure scalability of this process, data from previous tasks cannot be stored and hence is not available when learning a new task. A critical issue in such context, not addressed in the literature so far, relates to the decision of which expert to deploy at test time. We introduce a set of gating autoencoders that learn a representation for the task at hand, and, at test time, automatically forward the test sample to the relevant expert. This also brings memory efficiency as only one expert network has to be loaded into memory at any given time. Further, the autoencoders inherently capture the relatedness of one task to another, based on which the most relevant prior model to be used for training a new expert, with finetuning or learning without-forgetting, can be selected. We evaluate our method on image classification and video prediction problems.


This code contains 4 main files as an example on how to use Expert Gate: cnn_autoencoder_layer_relusig: trains a one layer autoencoder on a given task using adagrad (the code is in cnn_train_adagrad_oneLayer) compute_relatedness: compute relatedness between two tasks. Use it to select the most related previous task. You can also use the relatedness value to decide between finetuning and LwF as illustrated in our paper. Then the training of the expert model based on the most related task becomes standard transfer learning. Please contact me if you need help. test_triple_auto_gate: after training the autoencoders you can test the gate performance using this sample code. test_expret_networks_autoendoer_hard_gate: after training tasks autoencoders and expert models use this code to test the expert gate performance.
