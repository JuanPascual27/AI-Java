package SupervisedLearning.LinearRegression;

public class E2_CostFunction {
    public static void main(String[] args) {
        /*
        Like in exercise 1 we have the model of linear regression:
        f_wb(x) = wx + b, where w and b are called the model parameters
        also called coefficients or weights, and depending of this, we get
        different functions.}
        Also remember that in a linear function:
            w = slope of the straight line
            b = y-intercept

        An example data training is representing like (x^(i), y^(i)),
        but in a prediction we have a y-hat output that correspond to:
            y_hat = f_wb(x^(i)) = w * x^(i) + b
        in some cases there is a big difference between y^(i) and y_hat^(i).
        That is the cause why we need to find parameters w and b where y_hat^(i)
        is close to y^(i) for all (x^(i), y^(i)).

        To do that we have to construct the "cost function"/"squared error cost function"
        that compares the prediction y-hat with the real value y (y_hat^(i) - y^(i))^2
        this difference is called the error of the prediction and its squared to obtain
        positive values of all data training:

                   1     m
        J(w,b) = ---- *  E (y_hat^(i) - y^(i))^2
                  2m    i=1

        where:
            m = number of training examples

        Note:
            Divide by m to obtain the average of the error, and by convention its also divide
            by 2 but the last divide its no necessary.
            Also, in Machine Learning different persons use different cost function, but
            squared cost function its the most commonly used for linear regression.

                   1     m
        J(w,b) = ---- *  E (f_wb(x^(i)) - y^(i))^2
                  2m    i=1
         */

        /*
        Intuition of the Cost Function:

        Model:
            f_wb(x) = w * x + b
        Parameters:
            w, b
        Cost Function:
                       1     m
            J(w,b) = ---- *  E (f_wb(x^(i)) - y^(i))^2
                      2m    i=1
        Goal: (this is the goal of the linear regression)
            minimize J(w,b)
              w,b

        The cost function its a good approximation to know what are the best values to w and b
        comparing the result of the cost function that minimize its value (J(w,b))
         */

        //Training set x and y
        Double[] xTrain = {1.0, 2.0};
        Double[] yTrain = {300.0, 500.0};

        // Model parameters
        Double w = 100.0;
        Double b = 100.0;

        // Compute the cost function for parameters w = 100 and b = 100
        Double J_wb = computeCost(xTrain, yTrain, w, b);
        System.out.printf("J(%.2f, %.2f) = %.2f\n", w, b, J_wb);

        w = 190.0;
        b = 100.0;

        // Compute the cost function for parameters w = 190 and b = 100
        J_wb = computeCost(xTrain, yTrain, w, b);
        // And now we obtain better results (a minor cost/error)
        System.out.printf("J(%.2f, %.2f) = %.2f", w, b, J_wb);
    }

    public static Double computeCost(Double[] x, Double[] y, Double w, Double b) {
        /*
        Computes the cost function for linear regression.

        Args:
            x (array (m)): Data, m examples
            y (array (m)): target values
            w,b (scalar): model parameters

        Return:
            total_cost (double): The cost of using w,b as the parameters for linear regression
            to fit the data points in x and y
        */
        int m = x.length;
        double costSum = 0.0;

        for (int i = 0; i < m; i++) {
            Double yHat = w * x[i] + b;
            costSum += Math.pow((yHat - y[i]), 2.0);
        }
        return costSum / (2 * m);
    }
}
