use nalgebra::{Matrix3, UnitQuaternion, Vector3, Vector4};

/// A cascaded PID controller for quadcopter flight control.
///
/// This controller implements a hierarchical control architecture commonly used in
/// quadcopter autopilots. The control cascade consists of:
/// 1. Position control (outer loop) - generates desired accelerations
/// 2. Attitude control (middle loop) - converts accelerations to desired body rates
/// 3. Body rate control (inner loop) - generates motor torques
///
/// The controller handles 6-DOF flight control including position hold, trajectory tracking,
/// and attitude stabilization with configurable gains and physical constraints.
///
/// # Control Architecture
///
/// Position/Velocity Commands → Lateral Position Controller → Desired Accelerations
///                                                          ↓
/// Altitude Commands → Altitude Controller → Thrust + Roll/Pitch Controller → Body Rates
///                                                          ↓
/// Yaw Commands → Yaw Controller → Desired Yaw Rate
///                                 ↓
/// Body Rate Commands → Body Rate Controller → Moments → Motor Mixing → Motor Thrusts
///
pub struct QuadcopterController {
    /// Control loop time step (seconds)
    dt: f64,

    // Drone physical parameters
    /// Total mass of the quadcopter (kg)
    mass: f64,
    /// Moment of inertia about X-axis (body frame) (kg⋅m²)
    ixx: f64,
    /// Moment of inertia about Y-axis (body frame) (kg⋅m²)
    iyy: f64,
    /// Moment of inertia about Z-axis (body frame) (kg⋅m²)
    izz: f64,
    /// Maximum allowed tilt angle from vertical (radians)
    max_tilt_angle: f64,
    /// Maximum horizontal acceleration (m/s²)
    max_accel_xy: f64,
    /// Maximum horizontal velocity (m/s)
    max_vel_xy: f64,
    /// Maximum vertical velocity (m/s)
    max_ascent_rate: f64,
    /// Distance from center to motor (meters)
    l: f64,
    /// Minimum thrust per motor (N)
    min_motor_thrust: f64,
    /// Maximum thrust per motor (N)
    max_motor_thrust: f64,

    // Controller gains
    /// Proportional gains for body rate control [roll, pitch, yaw]
    kp_pqr: Vector3<f64>,
    /// Proportional gain for attitude (roll/pitch) stabilization
    kp_bank: f64,
    /// Proportional gain for altitude position control
    kp_pos_z: f64,
    /// Derivative gain for altitude velocity control
    kp_vel_z: f64,
    /// Integral gain for altitude position control
    ki_pos_z: f64,
    /// Proportional gain for horizontal position control
    kp_pos_xy: f64,
    /// Proportional gain for yaw angle control
    kp_yaw: f64,
    /// Derivative gain for horizontal velocity control
    kp_vel_xy: f64,
    /// Motor torque constant ratio (torque/thrust)
    kappa: f64,

    // Controller state
    /// Accumulated altitude error for integral control
    integrated_altitude_error: f64,
}

impl QuadcopterController {
    /// Creates a new quadcopter controller with specified physical parameters.
    ///
    /// This constructor initializes the controller with drone physical properties
    /// and operational limits. All control gains are initialized to zero and must
    /// be set using `set_gains()` before the controller can be used effectively.
    ///
    /// # Arguments
    ///
    /// * `dt` - Control loop time step in seconds (typically 0.001 to 0.02)
    /// * `mass` - Total quadcopter mass in kg
    /// * `ixx`, `iyy`, `izz` - Principal moments of inertia in kg⋅m²
    /// * `max_tilt_angle` - Maximum tilt from vertical in radians (safety limit)
    /// * `max_accel_xy` - Maximum horizontal acceleration in m/s² (performance limit)
    /// * `max_vel_xy` - Maximum horizontal velocity in m/s (performance limit)
    /// * `max_ascent_rate` - Maximum vertical velocity in m/s (performance limit)
    /// * `l` - Distance from center to motor in meters (arm length)
    /// * `min_motor_thrust` - Minimum thrust per motor in N (hardware limit)
    /// * `max_motor_thrust` - Maximum thrust per motor in N (hardware limit)
    ///
    /// # Returns
    ///
    /// A new `QuadcopterController` instance with gains initialized to zero
    ///
    /// # Example
    ///
    /// let controller = QuadcopterController::new(
    ///     0.01,  // 100 Hz control loop
    ///     1.5,   // 1.5 kg quadcopter
    ///     0.02, 0.02, 0.04,  // Moments of inertia
    ///     0.785, // 45 degree max tilt
    ///     10.0, 15.0, 5.0,   // Performance limits
    ///     0.25,  // 25 cm arm length
    ///     0.0, 8.0,  // Motor thrust range
    /// );
    ///
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

    /// Sets all controller gains for the cascaded control loops.
    ///
    /// This method configures the proportional, integral, and derivative gains
    /// for all control loops. Proper gain tuning is critical for stable flight.
    ///
    /// # Arguments
    ///
    /// * `kp_pqr` - Body rate P gains [roll_rate, pitch_rate, yaw_rate] (N⋅m⋅s/rad)
    /// * `kp_bank` - Attitude P gain for roll/pitch control (rad/s per rad)
    /// * `kp_pos_z` - Altitude position P gain (m/s² per m)
    /// * `kp_vel_z` - Altitude velocity D gain (m/s² per m/s)
    /// * `ki_pos_z` - Altitude position I gain (m/s² per m⋅s)
    /// * `kp_pos_xy` - Horizontal position P gain (m/s² per m)
    /// * `kp_yaw` - Yaw angle P gain (rad/s per rad)
    /// * `kp_vel_xy` - Horizontal velocity D gain (m/s² per m/s)
    /// * `kappa` - Motor torque-to-thrust ratio (dimensionless)
    ///
    /// # Tuning Guidelines
    ///
    /// Start with conservative gains and increase gradually:
    /// 1. Begin with body rate gains (innermost loop)
    /// 2. Tune attitude gains (middle loop)
    /// 3. Finally tune position gains (outer loop)
    ///
    /// # Example
    ///
    /// controller.set_gains(
    ///     Vector3::new(0.1, 0.1, 0.05),  // Body rate gains
    ///     5.0,   // Attitude gain
    ///     3.0, 6.0, 0.5,  // Altitude PID
    ///     2.0,   // Position gain
    ///     1.0,   // Yaw gain
    ///     4.0,   // Velocity gain
    ///     0.016, // Torque constant
    /// );
    ///
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

    /// Calculates desired 3-axis moments from body rate commands and measurements.
    ///
    /// This is the innermost control loop that directly controls the angular
    /// velocities of the quadcopter. It uses proportional control with inertia
    /// compensation to generate moments that will be converted to motor thrusts.
    ///
    /// The control law is: M = I * Kp * (ωcmd - ω)
    /// where I is the inertia matrix and Kp are the proportional gains.
    ///
    /// # Arguments
    ///
    /// * `pqr_cmd` - Desired body rates [p, q, r] in rad/s (body frame)
    /// * `pqr` - Current body rates [p, q, r] in rad/s (body frame)
    ///
    /// # Returns
    ///
    /// Desired moments [Mx, My, Mz] in N⋅m about body axes
    ///
    /// # Example
    ///
    /// let desired_rates = Vector3::new(0.1, -0.05, 0.0);  // Roll right, pitch down
    /// let current_rates = Vector3::new(0.05, -0.02, 0.01);
    /// let moments = controller.body_rate_control(desired_rates, current_rates);
    ///
    pub fn body_rate_control(&self, pqr_cmd: Vector3<f64>, pqr: Vector3<f64>) -> Vector3<f64> {
        let inertia = Matrix3::from_diagonal(&Vector3::new(self.ixx, self.iyy, self.izz));
        let pqr_err = pqr_cmd - pqr;
        let moment_cmd = (inertia * pqr_err).component_mul(&self.kp_pqr);

        moment_cmd
    }

    /// Calculates desired total thrust from altitude commands and current state.
    ///
    /// This function implements a PID controller for altitude hold and trajectory
    /// tracking. It accounts for the quadcopter's orientation to maintain proper
    /// vertical thrust regardless of tilt angle.
    ///
    /// The controller includes:
    /// - Proportional term: responds to position error
    /// - Derivative term: provides damping based on velocity error
    /// - Integral term: eliminates steady-state error
    /// - Feedforward term: compensates for desired acceleration
    ///
    /// # Arguments
    ///
    /// * `pos_z_cmd` - Desired altitude (m, positive up)
    /// * `vel_z_cmd` - Desired vertical velocity (m/s, positive up)
    /// * `pos_z` - Current altitude (m, positive up)
    /// * `vel_z` - Current vertical velocity (m/s, positive up)
    /// * `attitude` - Current attitude quaternion (inertial to body frame)
    /// * `accel_z_cmd` - Desired vertical acceleration feedforward (m/s², positive up)
    /// * `dt` - Time step for integral accumulation (seconds)
    ///
    /// # Returns
    ///
    /// Total thrust command in Newtons (positive up in body frame)
    ///
    /// # Notes
    ///
    /// The thrust is automatically limited based on motor constraints and ascent rate limits.
    /// The controller compensates for gravity and attitude changes automatically.
    ///
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

    /// Calculates desired body rates for roll and pitch attitude control.
    ///
    /// This function converts desired horizontal accelerations into roll and pitch
    /// body rate commands. It implements the middle loop of the control cascade,
    /// bridging between position control and body rate control.
    ///
    /// The controller:
    /// 1. Converts acceleration commands to desired tilt angles
    /// 2. Computes attitude errors in the body frame
    /// 3. Generates body rate commands to reduce attitude errors
    ///
    /// # Arguments
    ///
    /// * `accel_cmd` - Desired accelerations [ax, ay, az] in m/s² (inertial frame)
    /// * `attitude` - Current attitude quaternion (inertial to body frame)
    /// * `thrust` - Current total thrust command in N
    ///
    /// # Returns
    ///
    /// Desired body rates [p, q, r] in rad/s, where r (yaw rate) is set to 0
    ///
    /// # Control Law
    ///
    /// The desired tilt angles are: tan(φ) = ax/az, tan(θ) = ay/az
    /// Body rates are computed using the kinematic relationship between
    /// Euler angle rates and body rates, with proportional feedback.
    ///
    /// # Notes
    ///
    /// - Returns zero rates if thrust is zero (safety feature)
    /// - Tilt angles are limited by `max_tilt_angle` parameter
    /// - Only controls roll and pitch; yaw is handled separately
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

    /// Calculates desired horizontal accelerations from position and velocity commands.
    ///
    /// This function implements the outer loop of the position control cascade.
    /// It generates acceleration commands to track desired horizontal positions
    /// and velocities while incorporating feedforward acceleration terms.
    ///
    /// The control law combines:
    /// - Proportional feedback on position error
    /// - Derivative feedback on velocity error
    /// - Feedforward acceleration command
    ///
    /// # Arguments
    ///
    /// * `pos_cmd` - Desired position [x, y, z] in m (z component ignored)
    /// * `vel_cmd` - Desired velocity [vx, vy, vz] in m/s (z component ignored)
    /// * `pos` - Current position [x, y, z] in m
    /// * `vel` - Current velocity [vx, vy, vz] in m/s
    /// * `accel_cmd_ff` - Feedforward acceleration [ax, ay, az] in m/s² (z ignored)
    ///
    /// # Returns
    ///
    /// Desired acceleration command [ax, ay, 0] in m/s² (inertial frame)
    ///
    /// # Control Law
    ///
    /// a = Kp_pos * (pos_cmd - pos) + Kd_vel * (vel_cmd - vel) + accel_ff
    ///
    /// # Example
    ///
    /// let pos_cmd = Vector3::new(5.0, 3.0, 2.0);  // Desired position
    /// let vel_cmd = Vector3::new(1.0, 0.0, 0.0);  // Moving east at 1 m/s
    /// let current_pos = Vector3::new(4.5, 3.1, 2.0);
    /// let current_vel = Vector3::new(0.8, -0.1, 0.0);
    /// let feedforward = Vector3::zeros();
    ///
    /// let accel_cmd = controller.lateral_position_control(
    ///     pos_cmd, vel_cmd, current_pos, current_vel, feedforward
    /// );
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

    /// Calculates desired yaw rate from yaw angle command and current yaw.
    ///
    /// This function implements proportional control for yaw angle tracking.
    /// It converts yaw angle errors into yaw rate commands for the body rate
    /// controller to execute.
    ///
    /// # Arguments
    ///
    /// * `yaw_cmd` - Desired yaw angle in radians
    /// * `yaw` - Current yaw angle in radians
    ///
    /// # Returns
    ///
    /// Desired yaw rate in rad/s
    ///
    /// # Control Law
    ///
    /// ω_yaw = Kp_yaw * (yaw_cmd - yaw)
    ///
    /// # Notes
    ///
    /// This controller does not handle angle wrapping (±π transitions).
    /// For robust yaw control, consider implementing angle wrapping logic
    /// to handle the shortest angular path.
    ///
    /// # Example
    ///
    /// let desired_heading = std::f64::consts::PI / 4.0;  // 45 degrees
    /// let current_heading = 0.0;  // Facing north
    /// let yaw_rate = controller.yaw_control(desired_heading, current_heading);
    /// // Result: positive yaw rate to turn right

    pub fn yaw_control(&self, yaw_cmd: f64, yaw: f64) -> f64 {
        self.kp_yaw * (yaw_cmd - yaw)
    }

    /// Converts collective thrust and moment commands to individual motor thrust commands.
    ///
    /// This function implements the control allocation (motor mixing) for a quadcopter
    /// in X-configuration. It distributes the total thrust and three-axis moments
    /// across four motors to achieve the desired motion.
    ///
    /// # Motor Configuration (X-frame)
    ///   f2 (CCW)    f1 (CW)
    ///       \        /
    ///        \  +X  /
    ///         \    /
    ///    +Y    \  /
    ///           \/
    ///           /\
    ///          /  \
    ///         /    \
    ///        /      \
    ///   f3 (CW)    f4 (CCW)
    ///
    /// # Arguments
    ///
    /// * `coll_thrust_cmd` - Total thrust command in N (sum of all motors)
    /// * `moment_cmd` - Desired moments [Mx, My, Mz] in N⋅m about body axes
    ///
    /// # Returns
    ///
    /// Individual motor thrust commands [f1, f2, f3, f4] in N
    /// - f1: Front-left motor (CW rotation)
    /// - f2: Front-right motor (CCW rotation)
    /// - f3: Rear-left motor (CCW rotation)
    /// - f4: Rear-right motor (CW rotation)
    ///
    /// # Control Allocation
    ///
    /// The mixing matrix distributes:
    /// - Total thrust: equally among all four motors
    /// - Roll moment (Mx): differential thrust between left/right motor pairs
    /// - Pitch moment (My): differential thrust between front/rear motor pairs
    /// - Yaw moment (Mz): differential thrust between CW/CCW motor pairs
    ///
    /// # Example
    ///
    /// let total_thrust = 20.0;  // N (hover thrust for 2kg drone)
    /// let moments = Vector3::new(0.5, -0.3, 0.1);  // Roll right, pitch down, yaw left
    /// let motor_thrusts = controller.generate_motor_commands(total_thrust, moments);
    ///
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

    /// Main control loop that converts trajectory commands to motor thrust commands.
    ///
    /// This function implements the complete control pipeline from high-level
    /// trajectory commands (position, velocity, acceleration, attitude) to
    /// low-level motor thrust commands. It orchestrates all control loops in
    /// the proper sequence with appropriate limiting and safety checks.
    ///
    /// # Control Sequence
    ///
    /// 1. **Altitude Control**: Generate total thrust from vertical trajectory
    /// 2. **Position Control**: Generate horizontal acceleration commands
    /// 3. **Attitude Control**: Convert accelerations to body rate commands
    /// 4. **Yaw Control**: Generate yaw rate command from desired heading
    /// 5. **Body Rate Control**: Convert body rates to moments
    /// 6. **Motor Mixing**: Distribute thrust and moments to individual motors
    /// 7. **Safety Limiting**: Enforce motor thrust limits
    ///
    /// # Arguments
    ///
    /// * `t_pos` - Desired position [x, y, z] in m (inertial frame)
    /// * `t_vel` - Desired velocity [vx, vy, vz] in m/s (inertial frame)
    /// * `t_acc` - Desired acceleration [ax, ay, az] in m/s² (inertial frame)
    /// * `t_att` - Desired attitude quaternion (inertial to body frame)
    /// * `est_pos` - Current estimated position [x, y, z] in m
    /// * `est_vel` - Current estimated velocity [vx, vy, vz] in m/s
    /// * `est_omega` - Current estimated body rates [p, q, r] in rad/s
    /// * `est_att` - Current estimated attitude quaternion
    ///
    /// # Returns
    ///
    /// Motor thrust commands [f1, f2, f3, f4] in N, clamped to motor limits
    ///
    /// # Safety Features
    ///
    /// - Thrust limiting based on motor capabilities
    /// - Acceleration limiting for horizontal motion
    /// - Tilt angle limiting for attitude commands
    /// - Motor thrust clamping to prevent damage
    ///
    /// # Example
    ///
    /// // Hover at 5m altitude facing east
    /// let pos_cmd = Vector3::new(0.0, 0.0, 5.0);
    /// let vel_cmd = Vector3::zeros();
    /// let acc_cmd = Vector3::zeros();
    /// let att_cmd = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.0);
    ///
    /// // Current state (slightly off target)
    /// let est_pos = Vector3::new(0.1, -0.05, 4.95);
    /// let est_vel = Vector3::new(-0.02, 0.01, 0.03);
    /// let est_rates = Vector3::new(0.01, -0.005, 0.0);
    /// let est_att = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.02);
    ///
    /// let motor_cmds = controller.run_control(
    ///     pos_cmd, vel_cmd, acc_cmd, att_cmd,
    ///     est_pos, est_vel, est_rates, est_att
    /// );
    ///
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
