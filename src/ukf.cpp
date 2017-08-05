#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
//  P_ = MatrixXd(5, 5);
  P_ = Eigen::MatrixXd::Identity(5,5);
//  std::cout<<"P_"<<P_<<"\n";
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_=5;
  n_aug_= 7;
  n_z_=3;
  n_sig_=2 * n_aug_ + 1;
  lambda_=3-n_aug_;
  NIS_laser_=0.;
  NIS_radar_=0.;
  //* radar measurement dimension
  n_zlas_ = 2;
  weights_ = VectorXd(n_sig_);
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1; i<n_sig_; i++) {  //2n+1 weights
    weights_(i) = 0.5/(n_aug_+lambda_);
  }
  is_initialized_=false;

  Xsig_aug_=Eigen::MatrixXd(n_aug_,n_sig_);
  Xsig_pred_ = Eigen::MatrixXd(n_x_,n_sig_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(meas_package.sensor_type_==MeasurementPackage::LASER && !use_laser_) return;
  if(meas_package.sensor_type_==MeasurementPackage::RADAR && !use_radar_) return;

  cout<<"type:"<<meas_package.sensor_type_<<"\n";
    if(!is_initialized_){
      if(meas_package.sensor_type_==MeasurementPackage::LASER){
        x_(0)=meas_package.raw_measurements_(0); //px
        x_(1)=meas_package.raw_measurements_(1); //py
      }else if(meas_package.sensor_type_==MeasurementPackage::RADAR){
        float rho=meas_package.raw_measurements_[0];
        float phi=meas_package.raw_measurements_[1];
        float rho_dot=meas_package.raw_measurements_[2];
        x_(0) = cos(phi)*rho,
        x_(1) = sin(phi)*rho;
        x_(2) = abs(rho_dot);  // assuming speed of the rho be a initial of real speed
    }
    is_initialized_=true;

  }else{
    float delta_t=(meas_package.timestamp_ - time_us_) / 1000000.0;
    Prediction(delta_t);
    if(meas_package.sensor_type_==MeasurementPackage::LASER){
      UpdateLidar(meas_package);
    }else if(meas_package.sensor_type_==MeasurementPackage::RADAR){
      UpdateRadar(meas_package);
    }
  }
  time_us_=meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
//  GenerateSigmaPoints();
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_zlas_);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_zlas_,n_zlas_);

  // cross-correlation matrix Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_zlas_);

  // get predictions for x,S and Tc in Lidar space
  PredictLidarMeasurement(z_pred, S, Tc);

  // update the state using the LIDAR measurement
//  UpdateLidar(meas_package, z_pred, Tc, S);
  //mean predicted measurement
  VectorXd z = VectorXd::Zero(n_zlas_);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1);

  //Kalman gain K;
  MatrixXd K = MatrixXd::Zero(n_x_,n_zlas_);
  K = Tc * S.inverse();

  //residual
  VectorXd z_diff = VectorXd::Zero(n_zlas_);
  z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

  // update the time
//  previous_timestamp_ = meas_package.timestamp_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  You'll also need to calculate the radar NIS.
  */
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, n_sig_);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_,n_z_);

//  //transform sigma points into measurement space
//  int i;
//  double rho,phi_z,rho_dot;
//  double px,py,v,phi,phi_dot;
//  for(i=0;i<n_sig_;i++){
//    px=Xsig_pred_(0,i);
//    py=Xsig_pred_(1,i);
//    v=Xsig_pred_(2,i);
//    phi=Xsig_pred_(3,i);
//    phi_dot=Xsig_pred_(4,i);
//    rho=sqrt(px*px+py*py);
//    phi_z=atan2(py,px);
//    rho_dot=(px*cos(phi)*v+py*sin(phi)*v)/rho;
//    Zsig.col(i)<< rho,phi_z,rho_dot;
//  }
//  //calculate mean predicted measurement
//  z_pred.fill(0.);
//  for(i=0;i<n_sig_;i++){
//    z_pred+=weights_(i)*Zsig.col(i);
//  }
//  //calculate measurement covariance matrix S
//  S.fill(0.);
//  MatrixXd Zi;
//  for(i=0;i<n_sig_;i++){
//    Zi=Zsig.col(i)-z_pred;
//    while (Zi(1)> M_PI) Zi(1)-=2.*M_PI;
//    while (Zi(1)<-M_PI) Zi(1)+=2.*M_PI;
//
//    S+=weights_(i)*Zi*Zi.transpose();
//  }
//
//  MatrixXd R = MatrixXd(n_z_,n_z_);
//  R.fill(0.);
//  R(0,0)=std_radr_*std_radr_;
//  R(1,1)=std_radphi_*std_radphi_;
//  R(2,2)=std_radrd_*std_radrd_;
//  S+=R;

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    if(fabs(p_x) <= 0.0001){
      p_x = 0.0001;
    }
    if(fabs(p_y) <= 0.0001){
      p_y = 0.0001;
    }

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
//  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
//  MatrixXd S = MatrixXd(n_z_,n_z_);
  S.fill(0.0);

  //add measurement noise covariance matrix

  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  VectorXd z = VectorXd::Zero(3);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);
  //residual
  VectorXd z_diff =z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}



void UKF::AugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
//  Xsig_aug_.fill(0);
  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;

//  Eigen::MatrixXd sq=sqrt(lambda_+n_aug_) * L;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug_.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

}


void UKF::SigmaPointPrediction(double delta_t) {
  //predict sigma points
  Xsig_pred_.fill(0);

  for (int i = 0; i< n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

}

void UKF::PredictMeanAndCovariance() {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
  cout<<"x_after predict \n"<<x_<<"\n";
}


void UKF::PredictLidarMeasurement(VectorXd &z_out, MatrixXd &S_out, MatrixXd &Tc_out) {

  cout << "PredictLidarMeasurement \n";
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_zlas_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

    // measurement model
    Zsig(0,i) = Xsig_pred_(0,i);          //px
    Zsig(1,i) = Xsig_pred_(1,i);          //py

  }

  //mean predicted measurement
  static VectorXd z_pred = VectorXd(n_zlas_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  static MatrixXd S = MatrixXd(n_zlas_,n_zlas_);
  S.fill(0.0);

  //create matrix for cross correlation Tc
  static MatrixXd Tc = MatrixXd(n_x_, n_zlas_);
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

  }

  //add measurement noise covariance matrix
  static MatrixXd R = MatrixXd(n_zlas_,n_zlas_);
  R <<    pow(std_laspx_,2), 0,
  0, pow(std_laspy_,2);
  S = S + R;

  //write result
  z_out = z_pred;
  S_out = S;
  Tc_out = Tc;

  return;

}
