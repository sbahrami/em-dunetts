import numpy as np
import matplotlib.pyplot as plt

def perturb_last_dim(arr, delta):
    r_values = [np.random.uniform(0, delta) for i in range(arr.shape[-1])]
    for i in range(arr.shape[-1]):
        arr[..., i] = (arr[..., i] + r_values[i]) / (1 + sum(r_values))
        # normalize again
    return arr

def perturb_cpts(cpts, delta):
    perturbed_cpts = []
    delta1 = np.random.uniform(0, delta)
    delta2 = np.random.uniform(0, delta)
    for i in [0, 1, 2, 3, 4]:
        cpts[i] = perturb_last_dim(cpts[i], delta)
    return cpts

class EMLearner():
    def __init__(self):
        self.cpts = None

    def get_cpts(self):
        return self.cpts

    def eval_pr_table(self, cpts):
        # The join probability of Sloepnea,Foriennditis,Degar spots, Trimono-ht/s and dunetts syndrom
        f = cpts
        joint_pr = np.zeros(shape=(2, 2, 2, 2, 3))
        norm_joint_pr = np.zeros(shape=(2, 2, 2, 2, 3))
        for dune in range(3):
            for sloe in range(2):
                for fori in range(2):
                    for dega in range(2):
                        for trim in range(2):
                            # Evaluate joint probability P(dune, sloe, fori, dega, trim)
                            pr = (
                                f[0][dune] * f[1][dune, fori] * f[2][dune, dega] * 
                                f[3][trim] * f[4][dune, trim, sloe]
                            ).item()

                            if (pr > 1) or np.isnan(pr):
                                raise Exception("Wrong pr: {pr}")
                            joint_pr[sloe, fori, dega, trim, dune] = pr
        
        norm_joint_pr_den = np.expand_dims(joint_pr.sum(axis=-1), axis=-1)
        if norm_joint_pr_den.any() == 0:
            raise Exception("Zero norm_joint_pr_den")
        norm_joint_pr = joint_pr / norm_joint_pr_den
        # Replace nans with zeros
        norm_joint_pr = np.where(np.isnan(norm_joint_pr), 0, norm_joint_pr)

        # Ensure all probabilities are between 0 and 1
        if not np.all((joint_pr >= 0) & (joint_pr <= 1)):
            raise ValueError("Invalid probability values in joint_pr.")
        if not np.all((norm_joint_pr >= 0) & (norm_joint_pr <= 1)):
            raise ValueError("Invalid probability values in norm_joint_pr.")

        # Ensure the sum of joint probabilities is close to 1
        if not np.isclose(joint_pr.sum(), 1, atol=0.1):
            raise Warning(f"Sum of joint probabilities is {joint_pr.sum()}, but expected to be close to 1")
        
        return (joint_pr, norm_joint_pr)

    def expectation(self, data, joint_pr, norm_joint_pr):
        missing_count = (data[:, 4] == -1).sum()
        given_count = len(data) - missing_count
        new_data_count = missing_count * 3 + given_count
        idx = 0
        weighted_data = np.zeros(shape=(new_data_count, 7), dtype=np.float64)
        for example in data:
            weighted_example = np.zeros(shape=7, dtype=np.float64)

            # Estimate Dunetts if it's missing
            if example[4] == -1:
                for dune in range(3):
                    weighted_example[:4] = example[:4]
                    weighted_example[4] = dune
                    sloe, fori, dega, trim = example[0:4]
                    weighted_example[5] = joint_pr[int(sloe), int(fori), int(dega), int(trim), int(dune)]
                    if np.isnan(norm_joint_pr[int(sloe), int(fori), int(dega), int(trim), int(dune)]):
                        raise Exception("nan value")
                    weighted_example[6] = norm_joint_pr[int(sloe), int(fori), int(dega), int(trim), int(dune)]
                    weighted_data[idx, :] = weighted_example[:]
                    idx += 1
            else:
                weighted_example[:5] = example
                weighted_example[5:] = 1
                weighted_data[idx, :] = weighted_example
                idx += 1
        
        sum_joint_pr = weighted_data[:, 5].sum()
            
        return (weighted_data, sum_joint_pr)
    
    def maximization(self, weighted_data):
        cpts = [None] * 5

        # condition structure [sloe, fori, dega, trim, dune]
        # The key of each dictionary represents the index of the array to check the condition for
        # The values represent the value of that array index that need to be counted
        # The first dictionary is for numerator condition and the second for the denominator condition
        # factor_conditions = [
        #     [{4: [0, 1, 2]}, {}], # cpt_0
        #     [{1: [0, 1]}, {4: [0, 1, 2]}], # cpt_1
        #     [{2: [0, 1]}, {4: [0, 1, 2]}], # cpt_2
        #     [{3: [0, 1]}, {}], # cpt_3
        #     [{0: [0, 1]}, {4: [0, 1, 2], 3: [0, 1]}] # cpt_4
        # ]

        cpts[0] = np.array([weighted_data[weighted_data[:, 4] == i, 6].sum() for i in [0, 1, 2]])

        cpts[3] = np.array([weighted_data[weighted_data[:, 3] == i, 6].sum() for i in [0, 1]])

        prior_values = [0, 1, 2]
        cpt_prior = []
        for prior_value in prior_values:
            prior_data = (weighted_data[weighted_data[:, 4] == prior_value, :])
            cpt_prior.append(np.array([prior_data[prior_data[:, 1] == i, 6].sum() for i in [0, 1]]))
        cpts[1] = np.array(cpt_prior)

        prior_values = [0, 1, 2]
        cpt_prior = []
        for prior_value in prior_values:
            prior_data = (weighted_data[weighted_data[:, 4] == prior_value, :])
            cpt_prior.append(np.array([prior_data[prior_data[:, 2] == i, 6].sum() for i in [0, 1]]))
        cpts[2] = np.array(cpt_prior)

        prior_values_dune = [0, 1, 2]
        prior_values_trim = [0, 1]
        cpt_prior_dune = []
        for prior_value_dune in prior_values_dune:
            cpt_prior_trim = []
            prior_data_dune = (weighted_data[weighted_data[:, 4] == prior_value_dune, :])
            for prior_value_trim in prior_values_trim:
                prior_data = (prior_data_dune[prior_data_dune[:, 3] == prior_value_trim, :])
                cpt_prior_trim.append(np.array([prior_data[prior_data[:, 0] == i, 6].sum() for i in [0, 1]]))
            cpt_prior_dune.append(cpt_prior_trim)
        cpts[4] = np.array(cpt_prior_dune)

        for cpt in cpts:
            norm_cpt = np.sum(cpt, axis=-1, keepdims=True)
            cpt /= norm_cpt

        return cpts

    def fit(self, train_data: np.array, cpts, criteria=0.01, delta=1):
        for iter in range(2000):
            joint_pr, norm_joint_pr = self.eval_pr_table(cpts)
            weighted_data, new_sum_joint_pr = self.expectation(train_data, joint_pr, norm_joint_pr)
            cpts = self.maximization(weighted_data)
            if iter > 0:
                error_rate = abs(new_sum_joint_pr - prev_sum_joint_pr) / prev_sum_joint_pr
                # print(f"Error rate: {100*error_rate:.2f}%")
                if error_rate < criteria:
                    # print("EM solution converged successfully!")
                    break
            prev_sum_joint_pr = new_sum_joint_pr
        # if error_rate >= criteria:
            # print("The solution didn't converge!")
        self.cpts = cpts

    def predict(self, example: np.array, cpts):
        _, norm_joint_pr = self.eval_pr_table(cpts)
        example_to_int = list(example.astype(int))
        return np.argmax(norm_joint_pr[*example_to_int, :])

    def eval_accuracy(self, test_data, cpts):
        success = 0
        for test_data_point in test_data:
             test_features = test_data_point[:4]
             test_value = test_data_point[4]
             est_value = self.predict(test_features, cpts)
             if est_value == test_value:
                 success += 1
        return success/test_data.shape[0]

def read_data(file_name):
    data = np.genfromtxt(file_name)
    return data

def modify_data_type(data):
    # Example manipulation
    modified_data = data.copy()  # Copy to avoid modifying the original array

    # Perform your manipulations here...
    # Ensure that these manipulations do not change the type of the first 5 columns to non-integer

    # Cast the first 5 columns back to integers to ensure they remain integers
    modified_data[:, :5] = modified_data[:, :5].astype(np.int16)

    return modified_data

def main():
    # P(dune)
    # States: t->True, m->mild, s->severe, n-> not present
    cpt_0 = np.array([0.45, 0.27, 0.28])  # For dune = 'n', 'm', 's'. 

    # P(fori|dune) guessed piror
    cpt_1 = np.array([
        [0.9, 0.1],  # For dune='n'
        [0.1, 0.9],  # For dune='m'
        [0.95, 0.05]  # For dune='s'
    ])

    # P(dega|dune) guessed piror
    cpt_2 = np.array([
        [0.91, 0.09],  # For dune='n'
        [0.88, 0.02],  # For dune='m'
        [0.1, 0.9]  # For dune='s'
    ])

    # P(trim)
    cpt_3 = np.array([0.9, 0.1])

    # P(sloe|trim, dune) guessed piror
    cpt_4 = np.array([
        [[0.95, 0.05], [0.91, 0.09]],  # For dune='n'
        [[0.07, 0.93], [0.92, 0.08]],  # For dune='m'
        [[0.06, 0.94], [0.96, 0.04]]  # For dune='s'
    ])

    guess_cpts = [cpt_0, cpt_1, cpt_2, cpt_3, cpt_4]

    # Read train data
    train_data = read_data('./data/traindata.txt')
    test_data = read_data('./data/testdata.txt')

    em_learner = EMLearner()
    deltas = np.zeros(shape=20) 
    guess_accuracies = np.zeros(shape=(20, 20))
    em_test_accuracies = np.zeros(shape=(20, 20))
    for delta_idx, delta in enumerate(np.linspace(0, 4, num=20)):
        print(f"Delta: {delta:.2f}")
        for trial in range(20):
            print(f"Trial: {trial}")
            deltas[delta_idx] = delta
            perturbed_guess_cpts = perturb_cpts(guess_cpts, delta)
            em_learner.fit(train_data, perturbed_guess_cpts, criteria=0.01, delta=delta)
            em_cpts = em_learner.get_cpts()
            guess_accuracies[delta_idx, trial] = em_learner.eval_accuracy(test_data, perturbed_guess_cpts)
            em_test_accuracies[delta_idx, trial] = em_learner.eval_accuracy(test_data, em_cpts)
        print(f"mean: {em_test_accuracies[delta_idx, :].mean():.2f}, std: {em_test_accuracies[delta_idx, :].std():.2f}")

    _, ax = plt.subplots()
    ax.errorbar(deltas, guess_accuracies.mean(axis=1), yerr=guess_accuracies.std(axis=1), label='Guess VS Test Accuracy', alpha=0.5, fmt='-o', lw=0.5)
    ax.errorbar(deltas, em_test_accuracies.mean(axis=1), yerr=em_test_accuracies.std(axis=1), label='EM VS Test Accuracy', alpha=0.5, fmt='-o', lw=0.5)
    ax.set_title(f"Accuracy vs. Noise")
    ax.set_xlabel("Noise (Delta)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right', fontsize='small')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.show()

if __name__ == '__main__':
    main()