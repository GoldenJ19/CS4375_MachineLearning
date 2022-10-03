#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace std;

const string FileName = "titanic_project.csv", SEPARATOR = string(75, '-');
const int rowSize = 1047;

vector<vector<double>> ReadInputFile() {
    // Open input file & declare return vector
    ifstream input(FileName);
    vector<vector<double>> toReturn;
    string line;

    // Parse through input data, store columns into separate vectors
    vector<double> pclass, survived, sex, age;
    for( int i = 0; i < rowSize && input.good(); i++ ) {
        // Read data attribute
        getline(input, line);

        // Skip first line (column names)
        if( i == 0 ) {
            continue;
        }

        // Create stringstream & temp attribute vars
        stringstream ss(line);
        double attribute_pclass, attribute_survived, attribute_sex, attribute_age;
        string buf;

        // Parse through line...

        // ...for irrelevant attribute
        getline(ss, buf, ',');

        // ...for pclass attribute
        getline(ss, buf, ',');
        attribute_pclass = stod(buf);

        // ...for survived attribute
        getline(ss, buf, ',');
        attribute_survived = stod(buf);

        // ...for sex attribute
        getline(ss, buf, ',');
        attribute_sex = stod(buf);

        // ...for age attribute
        getline(ss, buf, ',');
        attribute_age = stod(buf);

        // Add attributes to corresponding vectors
        pclass.push_back(attribute_pclass);
        survived.push_back(attribute_survived);
        sex.push_back(attribute_sex);
        age.push_back(attribute_age);
    }

    // Add columns to return vector
    toReturn.push_back(pclass);
    toReturn.push_back(survived);
    toReturn.push_back(sex);
    toReturn.push_back(age);

    // Close input & return vector with columns
    input.close();
    return toReturn;
}

vector<double> sigmoid( vector<double> z ) {
    vector<double> v;
    for( int i = 0; i < z.size(); i++ ) {
        v.push_back(1.0 / ( 1+exp( -(z.at(i)) ) ));
    }
    return v;
}

/**
 *
 * @return exit codes: same as main
 */
int LRProgram() {
    /// Read data set
    vector<vector<double>> df = ReadInputFile();

    /// Grab columns from df
    vector<double> pclass = df.at(0);
    vector<double> survived = df.at(1); //survived is the target
    vector<double> sex = df.at(2); //sex is the predictor
    vector<double> age = df.at(3);

    // DEBUG: (print vector data read in from file)
    /*or( int j = 0; j < df.at(0).size(); j++ ) {
        for( int i = 0; i < df.size(); i++ ) {
            cout << df.at(i).at(j) << ' ';
        }
        cout << endl;
    }*/

    /// Create Logistic Regression Model

    //1. For the estimates use only the first 800 values, and for the test the rest of the values (So split into two new vectors)
    //2. find the mean for the predictor and the mean for target
    //3. Use the formula for estimate for w^
    //4. Use the result to estimate b^
    //5. Output coefficients
    //6. Use remaining data vectors to predict values
    //7. Write functions to calculate accuracy, sensitivity, specificity.
    //8. Output the test metrics and the run time for the algorithm. You can use chrono to
    //measure time. Measure just the training time of the algorithm.

    //Split vectors into train and test
    vector<double> train_survived(800);
    vector<double> test_survived;

    for(int i = 0; i < survived.size(); i++)
    {
        if(i < 800)
        {
            train_survived.at(i) = survived.at(i);
        }
        else
        {
            test_survived.push_back(survived.at(i));
        }
    }

    vector<double> train_sex(800);
    vector<double> test_sex;

    for(int i = 0; i < sex.size(); i++)
    {
        if(i < 800)
        {
            train_sex.at(i) = sex.at(i);
        }
        else
        {
            test_sex.push_back(sex.at(i));
        }
    }

    auto start = chrono::system_clock::now();
    /// Determine weights
    vector<double> weights = {1,1};
    double learning_rate = 0.001 ;

    // Build data matrix
    vector<vector<double>> data_matrix(2);
    for( int i = 0; i < 2; i++ ) {
        // Fill row
        data_matrix.at(i) = vector<double>(train_sex.size());
        for( int j = 0; j < train_sex.size(); j++ ) {
            if( i == 0 )
                data_matrix.at(i).at(j) = 1;
            else
                data_matrix.at(i).at(j) = train_sex.at(j);
        }
    }

    // Do iterations to build accurate weights
    // labels = train_survived ; no conversion needed due to nature of c++
    const int iterations = 50000; // iterations = { 5 * 10^x | x<=5, x∈Z+ }
    for( int i = 1; i <= iterations; i++ ) {
        // Update on progress
        if( i == 1 || i % (iterations / 20) == 0 )
            cout << "Iterating weights... (" << (i == 1 ? 0 : (double)i / iterations * 100) << "%)" << endl;

        // Matrix multiply data_matrix with weights
        vector<double> z;
        for( int j = 0; j < data_matrix.at(0).size(); j++ ) {
            z.push_back(data_matrix.at(0).at(j) * weights.at(0) + data_matrix.at(1).at(j) * weights.at(1));
        }

        // Calculate probability vector
        vector<double> prob_vector = sigmoid(z);

        // Calculate error
        vector<double> error;
        for( int j = 0; j < train_survived.size(); j++ ) {
            error.push_back(train_survived.at(j) - prob_vector.at(j));
        }

        // Calculate new weights
        vector<double> gradient = {0,0};
        for( int j = 0; j < error.size(); j++ ) {
            gradient.at(0) += data_matrix.at(0).at(j) * error.at(j);
            gradient.at(1) += data_matrix.at(1).at(j) * error.at(j);
        }
        vector<double> gradient_lr = {gradient.at(0)*learning_rate, gradient.at(1)*learning_rate};

        weights.at(0) += gradient_lr.at(0);
        weights.at(1) += gradient_lr.at(1);
    }
    cout << endl;

    // Print weights
    cout << "Weights:" << endl << endl;
    cout << "Intercept\t" << weights.at(0) << endl;
    cout << "sex1\t\t" << weights.at(1) << endl;
    cout << endl;

    auto end = chrono::system_clock::now();
    auto trainingTime = chrono::duration_cast<chrono::seconds>(end - start);

    /// Make Predictions
    // Build test matrix
    vector<vector<double>> test_matrix(2);
    for( int i = 0; i < 2; i++ ) {
        // Fill row
        test_matrix.at(i) = vector<double>(test_sex.size());
        for( int j = 0; j < test_sex.size(); j++ ) {
            if( i == 0 )
                test_matrix.at(i).at(j) = 1;
            else
                test_matrix.at(i).at(j) = test_sex.at(j);
        }
    }

    // Get log odds of test data
    // (test_matrix %*% weights = vector of n values)
    vector<double> logOdds;
    for( int j = 0; j < test_matrix.at(0).size(); j++ ) {
        logOdds.push_back(test_matrix.at(0).at(j) * weights.at(0) + test_matrix.at(1).at(j) * weights.at(1));
    }

    // Compute probabilities
    vector<double> probabilities;

    for(int i = 0; i < logOdds.size(); i++)
    {
        probabilities.push_back(exp(logOdds.at(i))/(1+exp(logOdds.at(i))));
    }

    vector<int> predictions;

    for(int i = 0; i < probabilities.size(); i++)
    {
        if(probabilities.at(i) >= 0.5)
        {
            predictions.push_back(1);
        }
        else
        {
            predictions.push_back(0);
        }
    }

    vector<vector<double>> confusion = {{0,0}, {0,0}};
    double accuracy = 0, sensitivity = 0, specificity = 0;

    // Build confusion matrix
    for( int i = 0; i < predictions.size(); i++ ) {
        confusion.at(predictions.at(i)).at(test_survived.at(i))++;  // Confusion Matrix
    }

    // Calculate other metrics
    accuracy = ( confusion.at(1).at(1) + confusion.at(0).at(0) ) / predictions.size();  // (TP + TN) / N
    sensitivity = confusion.at(1).at(1) / (confusion.at(1).at(1) + confusion.at(0).at(1)); // TP / (TP + FN)
    specificity = confusion.at(0).at(0) / (confusion.at(0).at(0) + confusion.at(1).at(0)); // TN / (TN + FP)

    //// Print Metrics
    cout << "Metrics:" << endl << endl;
    cout << "\tRef" << endl << "Pred\t0\t1" << endl;
    for( int i = 0; i < confusion.size(); i++ ) {
        cout << i << '\t';
        for( int j = 0 ; j < confusion.at(i).size(); j++) {
            cout << confusion.at(i).at(j) << '\t';
        }
        cout << endl;
    }
    cout << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;
    cout << "Training Time: " << trainingTime.count() << "s" <<  endl;
    cout << endl;

    //// Return exit code success
    cout << "Program operations complete. Exiting..." << endl;
    return 0;

    /// Return exit code success
    return 0;
}

double CalculateAgeLikelihood( double age, double mean, double variance ) {
    return 1 / sqrt( 2 * M_PI * variance ) * exp( -( pow((age-mean),2) / ( 2 * variance ) ) );
}

/**
 *
 * @return exit codes: same as main
 */
int NBProgram() {
    //// Read data set
    vector<vector<double>> df = ReadInputFile();

    // Grab columns from df
    vector<double> df_pclass = df.at(0);
    vector<double> df_survived = df.at(1);
    vector<double> df_sex = df.at(2);
    vector<double> df_age = df.at(3);

    // DEBUG: (print vector data read in from file)
    /*for( int j = 0; j < df.at(0).size(); j++ ) {
        for( int i = 0; i < df.size(); i++ ) {
            cout << df.at(i).at(j) << ' ';
        }
        cout << endl;
    }*/

    //// Split data into train/test
    vector<double> train_survived, train_age, train_pclass, train_sex,
            test_survived, test_age, test_pclass, test_sex;
    for(int i = 0; i < df_survived.size(); i++ ) {
        if( i < 800 ) {
            train_survived.push_back(df_survived.at(i));
            train_age.push_back(df_age.at(i));
            train_pclass.push_back(df_pclass.at(i));
            train_sex.push_back(df_sex.at(i));
        }
        else {
            test_survived.push_back(df_survived.at(i));
            test_age.push_back(df_age.at(i));
            test_pclass.push_back(df_pclass.at(i));
            test_sex.push_back(df_sex.at(i));
        }
    }

    //// Create Naive Bayes Model
    /// Begin measuring training time
    auto start = chrono::system_clock::now();

    /// Calculate coefficients: conditional probabilities
    // probability matrices
    vector<double> aPriori;
    vector<vector<double>> lh_age, lh_pclass, lh_sex;
    // sex, pclass, and age survival rates
    vector<double> sex0, sex1, pc1, pc2, pc3, mean_age, var_age;
    sex0 = sex1 = pc1 = pc2 = pc3 = mean_age = var_age = {0,0};
    // survival counts
    int surv0 = 0, surv1 = 0;

    // Determine survival rates for each variable
    for( int i = 0; i < train_survived.size(); i++ ) {
        // age (sum)
        if( train_survived.at(i) == 0 ) {
            mean_age.at(0) += train_age.at(i);
            surv0++;
        }
        else {
            mean_age.at(1) += train_age.at(i);
            surv1++;
        }

        // pclass (count)
        if( train_pclass.at(i) == 1 ) {
            if( train_survived.at(i) == 0 )
                pc1.at(0)++; // pclass 1 didn't survive
            else
                pc1.at(1)++; // pclass 1 did survive
        }
        else if(train_pclass.at(i) == 2) {
            if( train_survived.at(i) == 0 )
                pc2.at(0)++; // pclass 2 didn't survive
            else
                pc2.at(1)++; // pclass 2 did survive
        }
        else {
            if( train_survived.at(i) == 0 )
                pc3.at(0)++; // pclass 3 didn't survive
            else
                pc3.at(1)++; // pclass 3 did survive
        }

        // sex (count)
        if( train_sex.at(i) == 0 ) {
            if( train_survived.at(i) == 0 )
                sex0.at(0)++; // sex 0 didn't survive
            else
                sex0.at(1)++; // sex 0 did survive
        }
        else {
            if( train_survived.at(i) == 0 )
                sex1.at(0)++; // sex 1 didn't survive
            else
                sex1.at(1)++; // sex 1 did survive
        }
    }

    // Calculate mean of age in relation to survival
    mean_age.at(0) /= surv0;
    mean_age.at(1) /= surv1;

    // Calculate variance of age in relation to survival
    for( int i = 0; i < train_age.size(); i++ ) {
        if( train_survived.at(i) == 0 )
            var_age.at(0) += pow(train_age.at(i)-mean_age.at(0), 2);
        else
            var_age.at(1) += pow(train_age.at(i)-mean_age.at(1), 2);
    }
    var_age.at(0) /= surv0-1;
    var_age.at(1) /= surv1-1;

    // Put coefficients into coefficient vectors
    aPriori = { (sex0.at(0)+sex1.at(0))/train_survived.size(), (sex0.at(1)+sex1.at(1))/train_survived.size() };
    lh_age = {
            {0, mean_age.at(0), sqrt(var_age.at(0))},
            {1, mean_age.at(1), sqrt(var_age.at(1))}
    };
    lh_pclass = {
            {0, (pc1.at(0))/(pc1.at(0)+pc2.at(0)+pc3.at(0)), (pc2.at(0))/(pc1.at(0)+pc2.at(0)+pc3.at(0)), (pc3.at(0))/(pc1.at(0)+pc2.at(0)+pc3.at(0))},
            {1, (pc1.at(1))/(pc1.at(1)+pc2.at(1)+pc3.at(1)), (pc2.at(1))/(pc1.at(1)+pc2.at(1)+pc3.at(1)), (pc3.at(1))/(pc1.at(1)+pc2.at(1)+pc3.at(1))}
    };
    lh_sex = {
            {0, (sex0.at(0))/(sex0.at(0)+sex1.at(0)), (sex1.at(0))/(sex0.at(0)+sex1.at(0))},
            {1, (sex0.at(1))/(sex0.at(1)+sex1.at(1)), (sex1.at(1))/(sex0.at(1)+sex1.at(1))}
    };

    /// Stop measuring training time
    auto end = chrono::system_clock::now();
    auto trainingTime = chrono::duration_cast<chrono::nanoseconds>(end - start);

    /// Print coefficients
    cout << "Coefficients:" << endl << endl;

    // A-priori
    cout << "A-priori Probabilities:" << endl << "Y" << endl << "0\t1" << endl;
    for( double d : aPriori ) {
        cout << setprecision(5) << d << "\t";
    }
    cout << endl;
    // Conditionals
    cout << endl << "Conditional Probabilities:" << endl;
    // age
    cout << "age" << endl << "Y\t[,1]\t[,2]" << endl;
    for( vector<double>& v : lh_age ) {
        for( double d : v ) {
            cout << d << "\t";
        }
        cout << endl;
    }
    cout << endl;
    // pclass
    cout << "pclass" << endl << "Y\t1\t2\t3" << endl;
    for( vector<double>& v : lh_pclass ) {
        for( double d : v ) {
            cout << d << "\t";
        }
        cout << endl;
    }
    cout << endl;
    // sex
    cout << "sex" << endl << "Y\t0\t1" << endl;
    for( vector<double>& v : lh_sex ) {
        for( double d : v ) {
            cout << d << "\t";
        }
        cout << endl;
    }
    cout << endl;

    //// Make Predictions...
    vector<vector<double>> predictions_raw;
    vector<double> predictions;
    for( int i = 0; i < test_survived.size(); i++ ) {
        // initialize variables we'll need
        double age = test_age.at(i), pclass = test_pclass.at(i), sex = test_sex.at(i);
        double num_surv0 = 0, num_surv1 = 0, denom = 0;

        // Apply Bayes' Theorem (using 7_2_NBayes-scratch.Rmd as reference)
        num_surv0 = lh_pclass.at(0).at(pclass) * lh_sex.at(0).at(sex+1) * aPriori.at(0) * CalculateAgeLikelihood(age, mean_age.at(0), var_age.at(0));
        num_surv1 = lh_pclass.at(1).at(pclass) * lh_sex.at(1).at(sex+1) * aPriori.at(1) * CalculateAgeLikelihood(age, mean_age.at(1), var_age.at(1));
        denom = lh_pclass.at(1).at(pclass) * lh_sex.at(1).at(sex+1) * CalculateAgeLikelihood(age, mean_age.at(1), var_age.at(1)) * aPriori.at(1)
                + lh_pclass.at(0).at(pclass) * lh_sex.at(0).at(sex+1) * CalculateAgeLikelihood(age, mean_age.at(0), var_age.at(0)) * aPriori.at(0);

        // Determine prediction
        predictions_raw.push_back({num_surv0 / denom, num_surv1 / denom });
    }

    // Interpret probabilities, and make predictions_raw
    for( vector<double> v : predictions_raw ) {
        double prob_s = v.at(1);
        if(prob_s < 0.5 )   // probably didn't survive
            predictions.push_back(0);
        else                // probably did survive
            predictions.push_back(1);
    }

    /* DEBUG: (see predicted values)
    cout << "Predictions:" << endl;
    for( int i = 0; i < predictions.size(); i++ ) {
        cout << predictions.at(i) << " , " << test_survived.at(i) << endl;
    }*/

    //// Determine metrics
    vector<vector<double>> confusion = {{0,0}, {0,0}};
    double accuracy = 0, sensitivity = 0, specificity = 0;

    // Build confusion matrix
    for( int i = 0; i < predictions.size(); i++ ) {
        confusion.at(predictions.at(i)).at(test_survived.at(i))++;  // Confusion Matrix
    }

    // Calculate other metrics
    accuracy = ( confusion.at(1).at(1) + confusion.at(0).at(0) ) / predictions.size();  // (TP + TN) / N
    sensitivity = confusion.at(1).at(1) / (confusion.at(1).at(1) + confusion.at(0).at(1)); // TP / (TP + FN)
    specificity = confusion.at(0).at(0) / (confusion.at(0).at(0) + confusion.at(1).at(0)); // TN / (TN + FP)

    //// Print Metrics
    cout << "Metrics:" << endl << endl;
    cout << "\tRef" << endl << "Pred\t0\t1" << endl;
    for( int i = 0; i < confusion.size(); i++ ) {
        cout << i << '\t';
        for( int j = 0 ; j < confusion.at(i).size(); j++) {
            cout << confusion.at(i).at(j) << '\t';
        }
        cout << endl;
    }
    cout << endl;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << sensitivity << endl;
    cout << "Specificity: " << specificity << endl;
    cout << "Training Time: " << trainingTime.count() << "ns" <<  endl;
    cout << endl;

    //// Return exit code success
    cout << "Program operations complete. Exiting..." << endl;
    return 0;
}

/**
 *
 * @return exit codes: 0 = successful, 1 = input error, 2 = ...
 */
int main() {
    // Declare initial variables
    int choice;

    do {
        // Get program variation (1 = Logistic Regression, 2 = Naive Bayes)
        cout << "Linear Model Options:" << endl << "1. Logistic Regression" << endl << "2. Naive Bayes" << endl << endl;
        cout << "Select a linear model option (# from above) (0 to close):" << endl << ">";
        cin >> choice;
        cout << endl;

        if (choice == 1) {
            // Perform the Logistic Regression Program
            cout << "Selected Logistic Regression." << endl << endl;
            return LRProgram();
        } else if (choice == 2) {
            // Perform the Naive Bayes Program
            cout << "Selected Naive Bayes." << endl << endl;
            return NBProgram();
        } else if (choice == 0) {
            // Exit; as per request
            cout << "Exiting..." << endl;
            return 0;
        } else {
            // Repeat; invalid input
            cout << "Invalid input. Please re-enter." << endl << endl;
        }
    } while(choice < 0 || choice > 2);
}