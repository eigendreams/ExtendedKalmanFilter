#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
		MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {
	/**
  DONE:
	 * predict the state
	 */

	// I will have to assume that matrices were updated elsewhere!!!

	// As said in lesson:
	// So, for the prediction step, we can still use the regular Kalman filter equations
	// and the F matrix rather than the extended Kalman filter equations.
	// There is no input to consider right now!

	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	/**
  DONE:
	 * update the state by using Kalman Filter equations
	 */

	// This shall be the laser update!
	// So we can use matrices 'normally'

	// Since this is the linear filter case, it will correspond to
	// the laser measurement

	Eigen::VectorXd y_innovation   = z - H_ * x_;
	Eigen::MatrixXd S_weightdenom  = H_ * P_ * H_.transpose() + R_;
	Eigen::MatrixXd K_weightmatrix = P_ * H_.transpose() * S_weightdenom.inverse();

	x_ = x_ + K_weightmatrix * y_innovation;
	P_ = P_ - K_weightmatrix * H_ * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const MatrixXd &Hj, const MatrixXd &Rj) {
	/**
  DONE:
	 * update the state by using Extended Kalman Filter equations
	 */
#include <math.h>

	// map from cartesian to polar here
	Eigen::VectorXd h_x_(3);

	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);
	if ( abs(px) < 1e-3 && abs(py) < 1e-3 )
	{
		px = 1e-3 * copysign(1.0, px);
		py = 1e-3 * copysign(1.0, py);
	}

	h_x_(0) = sqrt( px * px + py * py );
	h_x_(1) = atan2( py, px );
	h_x_(2) = (px * vx + py * vy) / (h_x_(0));

	Eigen::VectorXd y_innovation   = z - h_x_;
	// limits check
	if ( y_innovation(1) > M_PI ) {
		while ( y_innovation(1) > M_PI )
			y_innovation(1) = y_innovation(1) - M_PI;
	}
	if ( y_innovation(1) < -M_PI) {
		while ( y_innovation(1) < -M_PI )
			y_innovation(1) = y_innovation(1) + M_PI;
		}

	Eigen::MatrixXd S_weightdenom  = Hj * P_ * Hj.transpose() + Rj;
	Eigen::MatrixXd K_weightmatrix = P_ * Hj.transpose() * S_weightdenom.inverse();

	x_ = x_ + K_weightmatrix * y_innovation;
	P_ = P_ - K_weightmatrix * Hj * P_;
}
