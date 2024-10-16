def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    Epsilon set to 1E-5 for convergance

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        def h(x, theta):
            z = np.dot(x, theta)
            return 1/(1 + np.exp(-z))

        def gradient_theta_j(x, y, theta):
            return -1/m * np.dot(x.T, (y-h(x, theta)))

        def hessian(x, theta):
            '''
            Reshape hypothesis function to (m,1)
            '''
            h_ = np.reshape(h(x,theta), (-1, 1))
            return 1/m * (np.dot(x.T, h_ * (1-h_) * x))

        def theta_update(x, y, theta):
            return theta - np.dot(np.linalg.inv(hessian(x, theta)), gradient_theta_j(x, y, theta))

        m, n = x.shape
        self.theta = np.zeros(n)
        self.results = np.array(self.theta)
        self.epsilon = 0.00001
        theta = self.theta
        theta_new = theta_update(x, y, theta)
        self.results = np.vstack([self.results, theta_new])
        max_iterations = 15
        i = 0
        while np.linalg.norm(theta_new - theta, 1) >= self.epsilon and i < max_iterations:
            self.results = np.vstack([self.results, theta_new])
            print(np.linalg.norm(theta_new - theta))
            i += 1
            theta = theta_new
            theta_new = theta_update(x, y, theta)

        self.theta = theta_new
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        return np.dot(x, self.theta) >= 0


        # *** END CODE HERE ***
