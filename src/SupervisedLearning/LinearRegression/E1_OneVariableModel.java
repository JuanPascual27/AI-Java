package SupervisedLearning.LinearRegression;

import java.util.Arrays;

public class E1_OneVariableModel {
    public static void main(String[] args) {
        /*
        Example:
            Predict the price of a house based on the house size.
            (This is a regression model cause we need to predict numbers "y"
            based in any value of "x" and infinitely many possible values/outputs,
            instead of classification model that predict classes/categories
            with a small number of possible outputs)
        Data:
            x/input variable/feature = size in feet^2
            y/output variable/target = price in $1000's

        Training Set: data used to train the model
            size in feet^2 | price in $1000's
         (1)      1.0      |      300.0
         (2)      2.0      |      500.0

        Extra notation:
            m = number of training example
            (x, y) = single training example
            (x^(i), y^(i)) = i^th training example
                            (1^st, 2^nd, ...)
                    (x^(1), y^(1)) = (1.0, 300.0)
                            x^(1) != x^1
         */

        /*
        Workflow:
            training set -> learning algorithm -> f (function/model)

            x (new input) -> f (model) -> y-hat (prediction/estimated y)

        In this problem we have:
            size -> f (model) -> price (estimated)
         */

        //Training set x and y
        Double[] x_train = {1.0, 2.0};
        Double[] y_train = {300.0, 500.0};

        /*
        We represent the function f in linear model with the linear function, that
        represent an approximate average that fit with the real values:
            y = mx + b, but in this case we use the notation
            f_wb(X) = f(x) = wx + b, where w and b are model parameter that describe the straight line

        Note: in some cases we can need a no linear function to fit data but in easy problems,
            linear function works better.

        In this case, we can resolve the problem with linear regression with one variable
        and one variable means that we have a single feature (x)
         */

        // Model parameters
        Double w = 100.0;
        Double b = 100.0;

        // Make predictions with model f_wb(x) = wx + b
        // In this case the model is f_wb(x) = (100)x + 100
        Double[] tmp_f_wb = computeModelOutput(x_train, w, b);
        System.out.printf("Predictions with model: f_wb(x) = %.2f * x + %.2f\n", w, b);
        System.out.println("Predicted: " + Arrays.toString(tmp_f_wb));
        // But, if compare with the real values this looks like the model
        // don't make good predictions
        System.out.println("Real: " + Arrays.toString(y_train));

        // If we adjust model parameters, we can obtain better predictions
        w = 200.0;
        b = 100.0;
        // Now the the model is f_wb(x) = (200)x + 100
        tmp_f_wb = computeModelOutput(x_train, w, b);
        System.out.printf("\nPredictions with model: f_wb(x) = %.2f * x + %.2f\n", w, b);
        System.out.println("Predicted: " + Arrays.toString(tmp_f_wb));
        // And we obtain better results
        System.out.println("Real: " + Arrays.toString(y_train));
    }

    public static Double[] computeModelOutput(Double[] x, Double w, Double b) {
        /*
          Computes the prediction of a linear model
          Args:
            x (array (m)): Data, m examples
            w,b (scalar)    : model parameters
          Returns
            f_wb (array (m)): model prediction
         */
        int m = x.length;
        Double[] f_wb = new Double[m];
        int i = 0;
        while (i < m) {
            f_wb[i] = w * x[i] + b;
            i++;
        }
        return f_wb;
    }
}
