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

	// There is an omission in the Udacity lesson, since this is a different
	// sensor and Jacobian, a different noise model for the sensor is ALSO
	// needed, otherwise matrix dimension will not even match

	// As you can see, this functions is clearly the same as in the normal update,
	// just providing masks for the H and R matrices, as I DO NOT want to modify them
	// for the non linear case (just to avoid constant modifications, besides, since the
	// linear case essentially leaves matrices unchanged, but the EKF has to calculate a new
	// Jacobian (Hj) at each step)

	Eigen::VectorXd y_innovation   = z - Hj * x_;
	Eigen::MatrixXd S_weightdenom  = Hj * P_ * Hj.transpose() + Rj;
	Eigen::MatrixXd K_weightmatrix = P_ * Hj.transpose() * S_weightdenom.inverse();

	x_ = x_ + K_weightmatrix * y_innovation;
	P_ = P_ - K_weightmatrix * Hj * P_;
}
