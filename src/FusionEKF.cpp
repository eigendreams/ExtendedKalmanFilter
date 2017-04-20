#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
				0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
				0, 0.0009, 0,
				0, 0, 0.09;

	/**
  DONE:
	 * Finish initializing the FusionEKF.
	 * Set the process and measurement noises
	 */

	// Just for initialization
	float dt  = 0.1f;
	float dt2 = dt * dt;

	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;

	// state vector has the form px, py, vx, vy
	// meas vector for radar has the form rho, phi, D(rho)

	// Naturally this one will be redefined at each step
	// assuming unitary (as in composed of only 1s) input
	// vector and null state vector (therefore Hj is full of 0s)

	Hj_ << 	0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0;

	// We will have to wait till the first measurement to init
	// the kalman filter fully though, in particular, if the first
	// measurment is from a radar, assume  for now a laser first
	// measurement

	Eigen::VectorXd x_init = Eigen::VectorXd(4);
	x_init << 0, 0, 0, 0;

	Eigen::MatrixXd P_init = MatrixXd(4, 4);
	P_init << 	1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1;
	P_init = 1e6 * P_init;

	// Later on we will need to recalculate this based on timestamps
	Eigen::MatrixXd F_init = MatrixXd(4, 4);
	F_init << 	1, 0, dt, 0,
				0, 1, 0, dt,
				0, 0, 1,  0,
				0, 0, 0,  1;

	// Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	Eigen::MatrixXd Qv_process_ = MatrixXd(2, 2);
	Qv_process_ <<	9, 0,
					0, 9;
	// Later on we will need to recalculate this based on timestamps
	Eigen::MatrixXd G_process_ = MatrixXd(4, 2);
	G_process_ <<	dt2 / 2.0f,          0,
					0,			dt2 / 2.0f,
					dt,					 0,
					0,					dt;

	Eigen::MatrixXd Q_process_ = G_process_ * Qv_process_ * G_process_.transpose();

	ekf_.Init(
			x_init,
			P_init,
			F_init,
			H_laser_,
			R_laser_,
			Q_process_
	);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	static double lastTimeStamp = 0;

	/*****************************************************************************
	 *  Initialization
	 ****************************************************************************/
	if (!is_initialized_) {
		/**
    DONE:
		 * Initialize the state ekf_.x_ with the first measurement.
		 * Create the covariance matrix.
		 * Remember: you'll need to convert radar from polar to cartesian coordinates.
		 */
		// first measurement
		cout << "EKF: " << endl;
		ekf_.x_ = VectorXd(4);
		ekf_.x_ << 1, 1, 1, 1;

		lastTimeStamp = measurement_pack.timestamp_ / 1.0e6;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			/**
      	  	  Convert radar from polar to cartesian coordinates and initialize state.
			 */

			float rho  = measurement_pack.raw_measurements_[0];
			float phi  = measurement_pack.raw_measurements_[1];
			float dphi = 0.0f; // we do not know this as radar loses information!
			float drho = measurement_pack.raw_measurements_[2];

			float  x = rho * cos(phi);
			float  y = rho * sin(phi);
			// Using the chain rule, hopefully the compiler ignores the right side
			float vx = drho * cos(phi) - rho * sin(phi) * dphi;
			float vy = drho * sin(phi) + rho * cos(phi) * dphi;

			// Set the state
			ekf_.x_ << x, y, vx, vy;
			// Init cov. as very first update with obv. 0 innovation since we just
			// set the state. No time dependant matrices (F and Q) are used for the
			// innovation step!
			// We will need to calculate a Jacobian though
			Hj_ = tools.CalculateJacobian(ekf_.x_);

			Eigen::VectorXd z_meas = Eigen::VectorXd(3);
			z_meas << rho, phi, drho;
			ekf_.UpdateEKF(z_meas, Hj_, R_radar_);
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			/**
      Initialize state.
			 */

			float  x = measurement_pack.raw_measurements_[0];
			float  y = measurement_pack.raw_measurements_[1];
			float vx = 0.0f;
			float vy = 0.0f;

			// Set the state
			ekf_.x_ << x, y, vx, vy;
			// Init cov. as very first update with obv. 0 innovation since we just
			// set the state. No time dependant matrices (F and Q) are used for the
			// innovation step!
			// The information form would be so much better for this!
			Eigen::VectorXd z_meas = Eigen::VectorXd(2);
			z_meas << x, y;
			ekf_.Update(z_meas);
		}

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	 *  Prediction
	 ****************************************************************************/

	/**
   DONE:
	 * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
	 * Update the process noise covariance matrix.
	 * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	 */

	// We will need to update F and Q based on elapsed time
	// Again the information form would be so much better for this!
	double currentTimeStamp = measurement_pack.timestamp_ / 1.0e6;
	float dt = (currentTimeStamp - lastTimeStamp);
	float dt2 = dt * dt;

	// Later on we will need to recalculate this based on timestamps
	Eigen::MatrixXd F_new = MatrixXd(4, 4);
	F_new << 	1, 0, dt, 0,
				0, 1, 0, dt,
				0, 0, 1,  0,
				0, 0, 0,  1;

	// Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	Eigen::MatrixXd Qv_process_ = MatrixXd(2, 2);
	Qv_process_ <<	9, 0,
					0, 9;
	// Later on we will need to recalculate this based on timestamps
	Eigen::MatrixXd G_process_ = MatrixXd(4, 2);
	G_process_ <<	dt2 / 2.0f,          0,
					0,			dt2 / 2.0f,
					dt,					 0,
					0,					dt;

	Eigen::MatrixXd Q_new_ = G_process_ * Qv_process_ * G_process_.transpose();

	// Set F and Q matrices
	ekf_.F_ = F_new;
	ekf_.Q_ = Q_new_;

	ekf_.Predict();

	/*****************************************************************************
	 *  Update
	 ****************************************************************************/

	/**
   DONE:
	 * Use the sensor type to perform the update step.
	 * Update the state and covariance matrices.
	 */

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		float rho  = measurement_pack.raw_measurements_[0];
		float phi  = measurement_pack.raw_measurements_[1];
		float drho = measurement_pack.raw_measurements_[2];

		// Set the state
		Eigen::VectorXd z_meas = Eigen::VectorXd(3);
		z_meas << rho, phi, drho;
		// Init cov. as very first update with obv. 0 innovation since we just
		// set the state. No time dependant matrices (F and Q) are used for the
		// innovation step!
		// We will need to calculate a Jacobian though
		Hj_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.UpdateEKF(z_meas, Hj_, R_radar_);
	} else {
		// Laser updates
		float  x = measurement_pack.raw_measurements_[0];
		float  y = measurement_pack.raw_measurements_[1];

		// Set the state
		Eigen::VectorXd z_meas = Eigen::VectorXd(2);
		z_meas << x, y;
		// Init cov. as very first update with obv. 0 innovation since we just
		// set the state. No time dependant matrices (F and Q) are used for the
		// innovation step!
		// The information form would be so much better for this!
		ekf_.Update(z_meas);
	}

	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
