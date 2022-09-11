/**
 * Name:            Justin Hardy
 * Class:           CS 4375.003 - Machine Learning
 * Professor:       Dr. Karen Mazidi
 *
 * File Name:       main.cpp
 * Dependencies:    Boston.csv
 *
 * Functions:       main()
 *                  SortVector(vector<typename T>)
 *                  PrintStats(vector<double>)
 *                  PrintCovariance(vector<double>, vector<double>, double, double)
 *                  PrintCorrelation(vector<double>, vector<double>, double, double, double)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <tgmath.h>

using namespace std;

const string inputFileName = "Boston.csv";
const int vecSize = 1000;

/**
 * Reused vector sort function created by me in CS 2337.
 * @tparam T
 * @param vect
 * @return reference to sorted vector
 */
template<typename T>
void SortVector(vector<T> &vect){
    bool changeMade;
    do {
        changeMade = false;
        for( int i = 0; i < vect.size()-1; i++ ) {
            if( vect[i] > vect[i+1] ) {
                T temp = vect[i];
                vect[i] = vect[i+1];
                vect[i+1] = temp;
                changeMade = true;
            }
        }
    } while(changeMade);
}

/**
 * @param data
 * @return vector with stats in the following order: sum, mean, median, lower range, upper range
 */
vector<double> PrintStats( vector<double> data ) {
    vector<double> sortedData(data.size());
    // Loop and gather info all-in-one
    double sum = 0, mean = 0 , median = 0, range_low = INT32_MAX, range_high = -1;
    for( int i = 0; i < data.size(); i++ ) {
        sum += data.at(i); // calculate sum
        range_low = data.at(i) < range_low ? data.at(i) : range_low; // calculate low range
        range_high = data.at(i) > range_high ? data.at(i) : range_high; // calculate high range
    }
    mean = sum / data.size(); // calculate mean

    // Insertion sort for median calculation
    for( int i = 0; i < data.size(); i++ ) {
        sortedData.at(i) = data.at(i); // copy vector data into to-be-sorted vector
    }
    SortVector(sortedData); // sort vector
    median = sortedData.size() % 2 == 0
            ? (sortedData.at(sortedData.size()/2) + sortedData.at(sortedData.size()/2)) / 2
            : sortedData.at(sortedData.size()/2); // calculate median

    /// 1. Print sum of numeric vector
    cout << "Sum: " << sum << endl;

    /// 2. Print mean of numeric vector
    cout << "Mean: " << mean << endl;

    /// 3. Print median of numeric vector
    cout << "Median: " << median << endl;

    /// 4. Print range of numeric vector
    cout << "Range: " << range_low << "-" << range_high << endl;

    // return statistics
    return vector<double>{sum, mean, median, range_low, range_high};
}

/**
 * @param rm
 * @param medv
 * @param mean_rm
 * @param mean_medv
 * @return covariance of rm and medv
 */
double PrintCovariance( vector<double> rm, vector<double> medv, double mean_rm, double mean_medv ) {
    /// 5. Covariance between rm and medv
    // calculate upper portion of covariance formula
    double upper = 0, lower = rm.size()-1;
    for( int i = 0; i < rm.size(); i++ ) {
        upper += ( rm[i] - mean_rm ) * ( medv[i] - mean_medv );
    }
    double covariance = upper / lower; // calculate covariance

    // print covariance
    cout << "Covariance: " << covariance << endl;

    // return statistic
    return covariance;
}

/**
 * @param rm
 * @param medv
 * @param mean_rm
 * @param mean_medv
 * @param covariance
 * @return correlation of rm and medv
 */
double PrintCorrelation( vector<double> rm, vector<double> medv, double mean_rm, double mean_medv, double covariance ) {
    /// 6. Correlation between rm and medv
    // calculate standard deviations
    double dev_rm = 0, dev_medv = 0;
    for( int i = 0; i < rm.size(); i++ ) {
        dev_rm +=  pow(rm[i] - mean_rm, 2);
        dev_medv +=  pow(medv[i] - mean_medv, 2);
    }
    dev_rm = sqrt(dev_rm/rm.size());
    dev_medv = sqrt(dev_medv/medv.size());

    double correlation = covariance / (dev_rm * dev_medv); // calculate correlation

    // print correlation
    cout << "Correlation: " << correlation << endl;

    // return statistic
    return correlation;
}

/**
 * @return exit codes: 0 = successful, 1 = file reading error
 */
int main() {
    /// Read in Boston.csv file
    // Declare file stream variables
    ifstream input;
    string line, input_rm, input_medv;

    // Open input file
    cout << "Opening input file, \"" << inputFileName << "\"." << endl;
    input.open(inputFileName);

    // Validate that input file opened
    if( !input.is_open() ) {
        cout << "Error opening the input file, \"" << inputFileName << "\"." << endl;
        return 1;
    }

    // Create input vectors
    vector<double> rm(vecSize), medv(vecSize);

    // Read input file stream, save into vectors
    cout << "Reading input file..." << endl;
    getline(input, line);

    // Output read heading
    cout << "\tinputted heading: " << line << endl;

    int observations = 0;
    for( ; input.good(); observations++ ) {
        // Read rm & medv values
        getline(input, input_rm, ',');
        getline(input, input_medv, '\n');

        // Parse to double & insert into vector
        rm.at(observations) = stof(input_rm);
        medv.at(observations) = stof(input_medv);
    }

    cout << "\tInput file successfully read. Observations: " << observations << "." << endl;

    // Resize vector to fit the number of read-in observations
    rm.resize(observations);
    medv.resize(observations);

    // Close file stream
    cout << "Closing file stream to \"" << inputFileName << "\"." << endl;
    input.close();

    /// Calculate statistics (based off of instructions)
    // Print general statistics
    cout << endl << "rm:" << endl;
    vector<double> stats_rm = PrintStats(rm);
    cout << endl << "medv:" << endl;
    vector<double> stats_medv = PrintStats(medv);
    cout << endl;

    // Print covariance & correlation statistics
    double covariance = PrintCovariance(rm, medv, stats_rm[1], stats_medv[1]);
    PrintCorrelation(rm, medv, stats_rm[1], stats_medv[1], covariance);

    /// Exit program
    cout << endl << "Programing operations complete. Exiting..." << endl;
    return 0;
}
