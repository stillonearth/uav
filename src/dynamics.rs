use fast_ode;
use nalgebra::Vector3;

/// Represents the complete state of a drone in 3D space.
///
/// This structure contains all the necessary state variables to describe a drone's
/// position, velocity, orientation (Euler angles), and angular velocities at any
/// given moment in time.
///
/// # Fields
///
/// * `position_x`, `position_y`, `position_z` - Position coordinates in the inertial frame (meters)
/// * `velocity_x`, `velocity_y`, `velocity_z` - Linear velocities in the inertial frame (m/s)
/// * `roll`, `pitch`, `yaw` - Euler angles representing orientation (radians)
/// * `roll_rate`, `pitch_rate`, `yaw_rate` - Angular velocities in the body frame (rad/s)
#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub position_x: f64,
    pub position_y: f64,
    pub position_z: f64,
    pub velocity_x: f64,
    pub velocity_y: f64,
    pub velocity_z: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub roll_rate: f64,
    pub pitch_rate: f64,
    pub yaw_rate: f64,
}

/// Physical constants and properties of the drone.
///
/// This structure contains the physical parameters needed for the drone dynamics
/// simulation, including gravitational acceleration, mass, and moments of inertia.
///
/// # Fields
///
/// * `g` - Gravitational acceleration (m/s²)
/// * `mass` - Total mass of the drone (kg)
/// * `ixx`, `iyy`, `izz` - Principal moments of inertia about body axes (kg⋅m²)
#[derive(Clone, Copy)]
pub struct Consts {
    pub g: f64,
    pub mass: f64,
    pub ixx: f64,
    pub iyy: f64,
    pub izz: f64,
}

impl State {
    /// Converts the state structure to a fixed-size array.
    ///
    /// This method is useful for interfacing with numerical integration libraries
    /// that expect state vectors as arrays.
    ///
    /// # Returns
    ///
    /// A 12-element array containing all state variables in the following order:
    /// [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    ///
    /// # Example
    ///
    /// ```rust
    /// let state = State::default();
    /// let array = state.to_array();
    /// assert_eq!(array.len(), 12);
    /// ```
    pub fn to_array(&self) -> [f64; 12] {
        [
            self.position_x as f64,
            self.position_y as f64,
            self.position_z as f64,
            self.velocity_x as f64,
            self.velocity_y as f64,
            self.velocity_z as f64,
            self.roll as f64,
            self.pitch as f64,
            self.yaw as f64,
            self.roll_rate as f64,
            self.pitch_rate as f64,
            self.yaw_rate as f64,
        ]
    }

    /// Creates a state structure from a fixed-size array.
    ///
    /// This method reconstructs a State from an array representation, typically
    /// used after numerical integration to convert back from array format.
    ///
    /// # Arguments
    ///
    /// * `arr` - A 12-element array containing state variables in the same order as `to_array()`
    ///
    /// # Returns
    ///
    /// A new State instance with values from the input array
    ///
    /// # Example
    ///
    /// ```rust
    /// let array = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0];
    /// let state = State::from_array(&array);
    /// assert_eq!(state.position_x, 1.0);
    /// ```
    pub fn from_array(arr: &[f64; 12]) -> Self {
        State {
            position_x: arr[0],
            position_y: arr[1],
            position_z: arr[2],
            velocity_x: arr[3],
            velocity_y: arr[4],
            velocity_z: arr[5],
            roll: arr[6],
            pitch: arr[7],
            yaw: arr[8],
            roll_rate: arr[9],
            pitch_rate: arr[10],
            yaw_rate: arr[11],
        }
    }
}

/// Ordinary Differential Equation (ODE) system for drone dynamics.
///
/// This structure implements the mathematical model for a quadrotor drone using
/// Newton's laws and Euler's equations of motion. It handles the transformation
/// between body-fixed and inertial reference frames.
///
/// # Fields
///
/// * `consts` - Physical constants and drone properties
/// * `forces` - Applied forces in the body frame (N)
/// * `torques` - Applied torques about body axes (N⋅m)
pub struct DroneOde {
    pub consts: Consts,
    pub forces: Vector3<f64>,
    pub torques: Vector3<f64>,
}

impl fast_ode::DifferentialEquation<12> for DroneOde {
    /// Computes the time derivatives of the state vector.
    ///
    /// This method implements the drone's equations of motion, including:
    /// - Translational dynamics using Newton's second law
    /// - Rotational dynamics using Euler's equations
    /// - Kinematic relationships between Euler angles and angular velocities
    /// - Body-to-inertial frame transformations
    ///
    /// # Arguments
    ///
    /// * `_t` - Current time (unused in this autonomous system)
    /// * `y` - Current state vector as a 12-element coordinate
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The state derivative vector (dy/dt)
    /// * A boolean indicating successful computation
    ///
    /// # Physics Implementation
    ///
    /// The method implements:
    /// 1. Position derivatives: ẋ = v (velocity)
    /// 2. Velocity derivatives: v̇ = F/m - g (Newton's second law with gravity)
    /// 3. Attitude derivatives: Euler angle kinematic equations
    /// 4. Angular velocity derivatives: Euler's rotational equations
    ///
    /// Special care is taken to avoid singularities at θ = ±π/2 (gimbal lock).
    fn ode_dot_y(&self, _t: f64, y: &fast_ode::Coord<12>) -> (fast_ode::Coord<12>, bool) {
        let state = y.0;

        // Extract state variables
        let phi = state[6]; // roll
        let theta = state[7]; // pitch
        let psi = state[8]; // yaw
        let p = state[9]; // roll rate
        let q = state[10]; // pitch rate
        let r = state[11]; // yaw rate

        // Trigonometric values
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let tan_theta = theta.tan();

        // Transform body forces to inertial frame using rotation matrix
        let cos_psi = psi.cos();
        let sin_psi = psi.sin();

        let fx_body = self.forces.x as f64;
        let fy_body = self.forces.y as f64;
        let fz_body = self.forces.z as f64;

        // Full rotation matrix transformation (Body to Inertial)
        // R = Rz(ψ) * Ry(θ) * Rx(φ)
        let fx_inertial = (cos_theta * cos_psi) * fx_body
            + (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * fy_body
            + (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * fz_body;

        let fy_inertial = (cos_theta * sin_psi) * fx_body
            + (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * fy_body
            + (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * fz_body;

        let fz_inertial = (-sin_theta) * fx_body
            + (sin_phi * cos_theta) * fy_body
            + (cos_phi * cos_theta) * fz_body;

        // State derivatives
        let mut dot_y = [0.0; 12];

        // Position derivatives (velocities)
        dot_y[0] = state[3]; // ẋ = vx
        dot_y[1] = state[4]; // ẏ = vy
        dot_y[2] = state[5]; // ż = vz

        // Velocity derivatives (accelerations)
        dot_y[3] = fx_inertial / self.consts.mass as f64; // v̇x = Fx/m
        dot_y[4] = fy_inertial / self.consts.mass as f64; // v̇y = Fy/m
        dot_y[5] = fz_inertial / self.consts.mass as f64 - self.consts.g as f64; // v̇z = Fz/m - g

        // Attitude derivatives (Euler angle rates)
        dot_y[6] = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta; // φ̇
        dot_y[7] = q * cos_phi - r * sin_phi; // θ̇
        dot_y[8] = if cos_theta.abs() > 1e-6 {
            q * sin_phi / cos_theta + r * cos_phi / cos_theta // ψ̇
        } else {
            0.0 // Avoid singularity at θ = ±π/2
        };

        // Angular velocity derivatives (Euler's equations)
        let ixx = self.consts.ixx as f64;
        let iyy = self.consts.iyy as f64;
        let izz = self.consts.izz as f64;

        dot_y[9] = (self.torques.x as f64 + (iyy - izz) * q * r) / ixx; // ṗ
        dot_y[10] = (self.torques.y as f64 + (izz - ixx) * r * p) / iyy; // q̇
        dot_y[11] = (self.torques.z as f64 + (ixx - iyy) * p * q) / izz; // ṙ

        (fast_ode::Coord(dot_y), true)
    }
}

/// Simulates drone dynamics over a specified time interval.
///
/// This function performs numerical integration of the drone's equations of motion
/// using an adaptive step-size ODE solver. It handles the complete 6-DOF dynamics
/// including translational and rotational motion.
///
/// # Arguments
///
/// * `initial_state` - Starting state of the drone
/// * `consts` - Physical parameters of the drone
/// * `forces` - Applied forces in body frame (N) - typically from propellers
/// * `torques` - Applied torques about body axes (N⋅m) - for attitude control
/// * `time_span` - Tuple (t_start, t_end) defining integration interval (seconds)
/// * `tolerance` - Absolute tolerance for the numerical integrator
///
/// # Returns
///
/// * `Ok(State)` - Final state after integration
/// * `Err(&str)` - Error message if integration fails
///
/// # Example
///
/// ```rust
/// let initial_state = State::default();
/// let consts = Consts { g: 9.81, mass: 1.0, ixx: 0.1, iyy: 0.1, izz: 0.2 };
/// let forces = Vector3::new(0.0, 0.0, 9.81); // Hover thrust
/// let torques = Vector3::zeros();
///
/// let result = simulate_drone(initial_state, consts, forces, torques, (0.0, 1.0), 1e-6);
/// ```
///
/// # Physics Notes
///
/// - Forces should be specified in the drone's body frame
/// - Positive Z-force typically represents upward thrust
/// - Gravity is automatically applied in the negative Z direction (inertial frame)
/// - The simulation accounts for gyroscopic effects and attitude-dependent force directions
pub fn simulate_drone(
    initial_state: State,
    consts: Consts,
    forces: Vector3<f64>,
    torques: Vector3<f64>,
    time_span: (f64, f64),
    tolerance: f64,
) -> Result<State, &'static str> {
    let ode = DroneOde {
        consts,
        forces,
        torques,
    };

    let initial_coord = fast_ode::Coord(initial_state.to_array());

    let result = fast_ode::solve_ivp(
        &ode,
        time_span,
        initial_coord,
        |_, _| true,
        tolerance,
        tolerance * 10.0,
    );

    match result {
        fast_ode::IvpResult::FinalTimeReached(final_coord) => Ok(State::from_array(&final_coord.0)),
        _ => Err("Integration failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests the drone's ability to maintain hover flight.
    ///
    /// This test verifies that when thrust equals weight and no torques are applied,
    /// the drone maintains its initial altitude and has minimal vertical velocity.
    #[test]
    fn test_hover_simulation() {
        let initial_state = State {
            position_x: 0.0,
            position_y: 0.0,
            position_z: 1.0,
            velocity_x: 0.0,
            velocity_y: 0.0,
            velocity_z: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            roll_rate: 0.0,
            pitch_rate: 0.0,
            yaw_rate: 0.0,
        };

        let consts = Consts {
            g: 9.81,
            mass: 1.0,
            ixx: 0.1,
            iyy: 0.1,
            izz: 0.2,
        };

        // Hover forces (thrust equals weight)
        let forces = Vector3::new(0.0, 0.0, consts.mass * consts.g);

        let torques = Vector3::new(0.0, 0.0, 0.0);

        let final_state =
            simulate_drone(initial_state, consts, forces, torques, (0.0, 1.0), 1e-6).unwrap();

        // In hover, position should remain approximately constant
        assert!((final_state.position_z - 1.0).abs() < 0.1);
        assert!(final_state.velocity_z.abs() < 0.1);
    }

    /// Tests free fall dynamics with no applied forces.
    ///
    /// This test verifies the gravitational acceleration is correctly implemented
    /// by comparing simulation results with analytical free fall equations.
    #[test]
    fn test_free_fall() {
        let initial_state = State {
            position_x: 0.0,
            position_y: 0.0,
            position_z: 10.0,
            velocity_x: 0.0,
            velocity_y: 0.0,
            velocity_z: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            roll_rate: 0.0,
            pitch_rate: 0.0,
            yaw_rate: 0.0,
        };

        let consts = Consts {
            g: 9.81,
            mass: 1.0,
            ixx: 0.1,
            iyy: 0.1,
            izz: 0.2,
        };

        // No forces (free fall)
        let forces = Vector3::zeros();

        let torques = Vector3::zeros();

        let t = 1.0;
        let final_state =
            simulate_drone(initial_state, consts, forces, torques, (0.0, t), 1e-6).unwrap();

        // Check free fall physics: z = z0 - 0.5*g*t^2
        let expected_z = 10.0 - 0.5 * 9.81 * t * t;
        let expected_vz = -9.81 * t;

        assert!((final_state.position_z - expected_z).abs() < 0.1);
        assert!((final_state.velocity_z - expected_vz).abs() < 0.1);
    }
}
