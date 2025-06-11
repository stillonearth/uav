use nalgebra::{Matrix3, UnitQuaternion, Vector3, Vector4};

pub struct QuadcopterController {
    dt: f64,

    // Drone physical parameters
    mass: f64,
    ixx: f64,
    iyy: f64,
    izz: f64,
    max_tilt_angle: f64,
    max_accel_xy: f64,
    max_vel_xy: f64,
    max_ascent_rate: f64,
    l: f64,
    min_motor_thrust: f64,
    max_motor_thrust: f64,

    // Controller gains
    kp_pqr: Vector3<f64>,
    kp_bank: f64,
    kp_pos_z: f64,
    kp_vel_z: f64,
    ki_pos_z: f64,
    kp_pos_xy: f64,
    kp_yaw: f64,
    kp_vel_xy: f64,
    kappa: f64,

    // Controller state
    integrated_altitude_error: f64,
}

impl QuadcopterController {
    pub fn new(
        dt: f64,
        mass: f64,
        ixx: f64,
        iyy: f64,
        izz: f64,
        max_tilt_angle: f64,
        max_accel_xy: f64,
        max_vel_xy: f64,
        max_ascent_rate: f64,
        l: f64,
        min_motor_thrust: f64,
        max_motor_thrust: f64,
    ) -> Self {
        let controller = Self {
            dt,
            mass,
            ixx,
            iyy,
            izz,
            max_tilt_angle,
            max_accel_xy,
            max_vel_xy,
            max_ascent_rate,
            l,
            min_motor_thrust,
            max_motor_thrust,
            kp_pqr: Vector3::new(0.0, 0.0, 0.0),
            kp_bank: 0.0,
            kp_pos_z: 0.0,
            kp_vel_z: 0.0,
            ki_pos_z: 0.0,
            kp_pos_xy: 0.0,
            kp_yaw: 0.0,
            kp_vel_xy: 0.0,
            kappa: 1.0,
            integrated_altitude_error: 0.0,
        };

        controller
    }

    pub fn set_gains(
        &mut self,
        kp_pqr: Vector3<f64>,
        kp_bank: f64,
        kp_pos_z: f64,
        kp_vel_z: f64,
        ki_pos_z: f64,
        kp_pos_xy: f64,
        kp_yaw: f64,
        kp_vel_xy: f64,
        kappa: f64,
    ) {
        self.kp_pqr = kp_pqr;
        self.kp_bank = kp_bank;
        self.kp_pos_z = kp_pos_z;
        self.kp_vel_z = kp_vel_z;
        self.ki_pos_z = ki_pos_z;
        self.kp_pos_xy = kp_pos_xy;
        self.kp_yaw = kp_yaw;
        self.kp_vel_xy = kp_vel_xy;
        self.kappa = kappa;
    }

    /// Calculate a desired 3-axis moment given a desired and current body rate
    pub fn body_rate_control(&self, pqr_cmd: Vector3<f64>, pqr: Vector3<f64>) -> Vector3<f64> {
        let inertia = Matrix3::from_diagonal(&Vector3::new(self.ixx, self.iyy, self.izz));
        let pqr_err = pqr_cmd - pqr;
        let moment_cmd = (inertia * pqr_err).component_mul(&self.kp_pqr);

        moment_cmd
    }

    /// Calculate desired quad thrust based on altitude setpoint and current state
    pub fn altitude_control(
        &mut self,
        pos_z_cmd: f64,
        vel_z_cmd: f64,
        pos_z: f64,
        vel_z: f64,
        attitude: UnitQuaternion<f64>,
        accel_z_cmd: f64,
        dt: f64,
    ) -> f64 {
        let pos_z_err = pos_z_cmd - pos_z;
        let vel_z_err = vel_z_cmd - vel_z;
        self.integrated_altitude_error += pos_z_err * dt;

        // Get rotation matrix from quaternion to extract b_z component
        let rotation_matrix = attitude.to_rotation_matrix();
        let b_z = rotation_matrix[(2, 2)];

        let p_term = self.kp_pos_z * pos_z_err;
        let d_term = self.kp_vel_z * vel_z_err;
        let i_term = self.ki_pos_z * self.integrated_altitude_error;

        let u1_bar = p_term + i_term + d_term + accel_z_cmd;

        let mut acc = (u1_bar + 9.81) / b_z;

        // Clip acceleration
        let max_acc = self.max_ascent_rate / dt;
        acc = acc.clamp(-max_acc, max_acc);

        self.mass * acc
    }

    /// Calculate desired body rates for roll and pitch control
    pub fn roll_pitch_control(
        &self,
        accel_cmd: Vector3<f64>,
        attitude: UnitQuaternion<f64>,
        thrust: f64,
    ) -> Vector3<f64> {
        let mut pqr_cmd = Vector3::zeros();

        if thrust == 0.0 {
            return pqr_cmd;
        }

        let coll_accel = -thrust / self.mass;

        let bx_cmd = (accel_cmd[0] / coll_accel).clamp(-self.max_tilt_angle, self.max_tilt_angle);
        let by_cmd = (accel_cmd[1] / coll_accel).clamp(-self.max_tilt_angle, self.max_tilt_angle);

        // Get rotation matrix from quaternion
        let rotation_matrix = attitude.to_rotation_matrix();
        let r = rotation_matrix.matrix();

        let bx_err = bx_cmd + r[(0, 2)];
        let by_err = by_cmd + r[(1, 2)];

        let bx_p_term = self.kp_bank * bx_err;
        let by_p_term = self.kp_bank * by_err;

        let r11 = r[(1, 0)] / r[(2, 2)];
        let r12 = -r[(0, 0)] / r[(2, 2)];
        let r21 = r[(1, 1)] / r[(2, 2)];
        let r22 = -r[(0, 1)] / r[(2, 2)];

        pqr_cmd[0] = r11 * bx_p_term + r12 * by_p_term;
        pqr_cmd[1] = r21 * bx_p_term + r22 * by_p_term;

        pqr_cmd
    }

    /// Calculate desired horizontal acceleration based on position/velocity commands
    pub fn lateral_position_control(
        &self,
        mut pos_cmd: Vector3<f64>,
        mut vel_cmd: Vector3<f64>,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
        mut accel_cmd_ff: Vector3<f64>,
    ) -> Vector3<f64> {
        // Zero out Z components
        accel_cmd_ff[2] = 0.0;
        vel_cmd[2] = 0.0;
        pos_cmd[2] = pos[2];

        let mut accel_cmd = accel_cmd_ff;

        let pos_err = pos_cmd - pos;
        let vel_err = vel_cmd - vel;

        let accel = self.kp_pos_xy * pos_err + self.kp_vel_xy * vel_err + accel_cmd_ff;
        accel_cmd[0] = accel[0];
        accel_cmd[1] = accel[1];

        accel_cmd
    }

    /// Calculate desired yaw rate to control yaw
    pub fn yaw_control(&self, yaw_cmd: f64, yaw: f64) -> f64 {
        self.kp_yaw * (yaw_cmd - yaw)
    }

    /// Convert collective thrust and moment commands to individual motor thrusts
    pub fn generate_motor_commands(
        &self,
        coll_thrust_cmd: f64,
        moment_cmd: Vector3<f64>,
    ) -> Vector4<f64> {
        let l = self.l / 2.0_f64.sqrt();

        let t1 = moment_cmd[0] / l;
        let t2 = moment_cmd[1] / l;
        let t3 = -moment_cmd[2] / self.kappa;
        let t4 = coll_thrust_cmd;

        Vector4::new(
            (t1 + t2 + t3 + t4) / 4.0,  // front left  - f1
            (-t1 + t2 - t3 + t4) / 4.0, // front right - f2
            (t1 - t2 - t3 + t4) / 4.0,  // rear left   - f4
            (-t1 - t2 + t3 + t4) / 4.0, // rear right  - f3
        )
    }

    /// Main control loop - converts trajectory commands to motor commands
    pub fn run_control(
        &mut self,
        t_pos: Vector3<f64>,
        t_vel: Vector3<f64>,
        t_acc: Vector3<f64>,
        t_att: UnitQuaternion<f64>,
        est_pos: Vector3<f64>,
        est_vel: Vector3<f64>,
        est_omega: Vector3<f64>,
        est_att: UnitQuaternion<f64>,
    ) -> Vector4<f64> {
        // Altitude control
        let mut thrust = self.altitude_control(
            t_pos[2], t_vel[2], est_pos[2], est_vel[2], est_att, t_acc[2], self.dt,
        );

        // Clip thrust to motor limits
        let thrust_margin = 0.1 * (self.max_motor_thrust - self.min_motor_thrust);
        thrust = thrust.clamp(
            (self.min_motor_thrust + thrust_margin) * 4.0,
            (self.max_motor_thrust - thrust_margin) * 4.0,
        );

        // Lateral position control
        let mut des_acc = self.lateral_position_control(t_pos, t_vel, est_pos, est_vel, t_acc);

        // Clip lateral accelerations
        des_acc[0] = des_acc[0].clamp(-self.max_accel_xy, self.max_accel_xy);
        des_acc[1] = des_acc[1].clamp(-self.max_accel_xy, self.max_accel_xy);

        // Roll-pitch control
        let mut des_omega = self.roll_pitch_control(des_acc, est_att, thrust);

        // Yaw control
        let traj_yaw = extract_yaw_from_quaternion(&t_att);
        let est_yaw = extract_yaw_from_quaternion(&est_att);
        des_omega[2] = self.yaw_control(traj_yaw, est_yaw);

        // Body rate control
        let des_moment = self.body_rate_control(des_omega, est_omega);

        // Generate motor commands
        let motors = self.generate_motor_commands(thrust, des_moment);

        // Clip motor commands to physical limits
        Vector4::new(
            motors[0].clamp(self.min_motor_thrust, self.max_motor_thrust),
            motors[1].clamp(self.min_motor_thrust, self.max_motor_thrust),
            motors[2].clamp(self.min_motor_thrust, self.max_motor_thrust),
            motors[3].clamp(self.min_motor_thrust, self.max_motor_thrust),
        )
    }
}

/// Extract yaw angle from quaternion (ZYX Euler convention)
fn extract_yaw_from_quaternion(quaternion: &UnitQuaternion<f64>) -> f64 {
    quaternion.euler_angles().2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = QuadcopterController::new(
            0.01, // dt
            0.5,  // mass
            0.01, // Ixx
            0.01, // Iyy
            0.02, // Izz
            0.5,  // max_tilt_angle
            5.0,  // max_accel_xy
            10.0, // max_vel_xy
            5.0,  // max_ascent_rate
            0.3,  // l
            0.1,  // min_motor_thrust
            2.0,  // max_motor_thrust
        );

        assert_eq!(controller.mass, 0.5);
        assert_eq!(controller.dt, 0.01);
    }

    #[test]
    fn test_body_rate_control() {
        let mut controller = QuadcopterController::new(
            0.01, 0.5, 0.01, 0.01, 0.02, 0.5, 5.0, 10.0, 5.0, 0.3, 0.1, 2.0,
        );

        // Set non-zero gains for the test
        controller.set_gains(
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
            1.0,
            1.0,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
        );

        let pqr_cmd = Vector3::new(0.1, 0.0, 0.0);
        let pqr = Vector3::new(0.0, 0.0, 0.0);

        let moment = controller.body_rate_control(pqr_cmd, pqr);

        // Should have non-zero moment in roll axis
        assert!(moment[0] != 0.0);
        assert_eq!(moment[1], 0.0);
        assert_eq!(moment[2], 0.0);
    }

    #[test]
    fn test_yaw_extraction() {
        // Identity quaternion should give 0 yaw
        let identity_quat = UnitQuaternion::identity();
        let yaw = extract_yaw_from_quaternion(&identity_quat);
        assert!((yaw - 0.0).abs() < 1e-10);

        // Test with 90 degree yaw rotation
        let yaw_90_quat =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::PI / 2.0);
        let yaw_90 = extract_yaw_from_quaternion(&yaw_90_quat);
        assert!(
            (yaw_90 - std::f64::consts::PI / 2.0).abs() < 1e-6,
            "Expected {}, got {}",
            std::f64::consts::PI / 2.0,
            yaw_90
        );

        // Test with -90 degree yaw rotation
        let yaw_neg90_quat =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -std::f64::consts::PI / 2.0);
        let yaw_neg90 = extract_yaw_from_quaternion(&yaw_neg90_quat);
        assert!(
            (yaw_neg90 + std::f64::consts::PI / 2.0).abs() < 1e-6,
            "Expected {}, got {}",
            -std::f64::consts::PI / 2.0,
            yaw_neg90
        );
    }
}
