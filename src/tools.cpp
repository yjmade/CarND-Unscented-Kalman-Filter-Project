#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
  if(estimations.size()==0 or estimations.size()!=ground_truth.size()){
    cout<< "Invalid Input"<<endl;
    return rmse;
  }
  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    rmse =rmse+ (ground_truth[i]-estimations[i]).array().pow(2).matrix();

  }

  //calculate the mean
  rmse=rmse.array()/estimations.size();
  //calculate the squared root
  rmse=rmse.array().sqrt();
  //return the result
  return rmse;
}
