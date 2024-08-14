This code trains an MLP on HCP dataset (cotains 1000 trials, 100 subjects, 360 brain regions).
 At each round of training one brain region is removed and the network is trained with the rest (e.g., in round1,
 359 ROIs are used as input and we trained 360 models in total).
 In the last round, we are left out with 8 brain regions that significantly contributed to the accuracy of our NN).
