### Explanation of Code

This code implements an Expectation-Maximization (EM) algorithm for learning conditional probability tables (CPTs) in a Bayesian network model. The Bayesian network represents dependencies between variables in a probabilistic graphical model. The problem is to learn a model that can predict Dunetts Syndrome based on a set of evidence including symptoms or the presence of a gene.

#### Key Functions:

1. **`perturb_last_dim(arr, delta)`**:
   - This function perturbs the values of the last dimension of a numpy array `arr` by adding random noise.
   - `delta` specifies the maximum amplitude of the perturbation.

2. **`perturb_cpts(cpts, delta)`**:
   - This function perturbs the conditional probability tables (CPTs) by calling `perturb_last_dim` for each dimension.
   - It returns the perturbed CPTs.

3. **`EMLearner` Class**:
   - Implements the EM learning algorithm for Bayesian networks.
   - `fit(train_data, cpts, criteria, delta)`: Fits the model to training data using the EM algorithm.
   - `eval_pr_table(cpts)`: Evaluates the joint and normalized joint probabilities based on the current CPTs.
   - `expectation(data, joint_pr, norm_joint_pr)`: Performs the expectation step of the EM algorithm.
   - `maximization(weighted_data)`: Performs the maximization step of the EM algorithm to update CPTs.
   - `predict(example, cpts)`: Predicts the missing value in an example using the learned CPTs.
   - `eval_accuracy(test_data, cpts)`: Evaluates the accuracy of the learned model on test data.

4. **`read_data(file_name)` and `modify_data_type(data)`**:
   - Helper functions to read and modify the dataset.
   - `modify_data_type` ensures the data types remain consistent.

5. **`main()`**:
   - Reads train and test data.
   - Initializes initial CPTs.
   - Iterates over noise levels (`delta`) and trials, evaluating accuracy.
   - Plots accuracy vs. noise for both original guess and EM-learned models.

#### Usage:
- Run the `main()` function to execute the EM algorithm and visualize the results.
- Ensure data files (`traindata.txt`, `testdata.txt`) are present in the `./data` directory.
- Adjust parameters such as `criteria` and `delta` for model fitting as needed.
