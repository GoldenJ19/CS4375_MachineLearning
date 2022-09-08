/**
 * Name:            Justin Hardy
 * Class:           CS 4375.003 - Machine Learning
 * Professor:       Dr. Karen Mazidi
 *
 * File Name:       main.cpp
 * Dependencies:    Boston.csv
 *
 * Functions:       main()
 *                  PrintStats(vector<double>)
 *                  PrintCovariance(vector<double>, vector<double>)
 *                  PrintCorrelation(vector<double>, vector<double>)
 */

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

const string inputFileName = "Boston.csv";
const int vecSize = 1000;

bool PrintStats( vector<double> data ) {
    // 1. Sum of numeric vector


    // 2. Mean of numeric vector


    // 3. Median of numeric vector


    // 4. Range of numeric vector


}

bool PrintCovariance( vector<double> rm, vector<double> medv ) {
    // 5. Covariance between rm and medv


}

bool PrintCorrelation( vector<double> rm, vector<double> medv ) {
    // 6. Correlation between rm and medv


}

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
    PrintStats(rm);
    PrintStats(medv);

    // Print covariance & correlation statistics
    PrintCovariance(rm, medv);
    PrintCorrelation(rm, medv);

    // Exit program
    cout << endl << "Programing operations complete. Exiting..." << endl;
    return 0;
}
