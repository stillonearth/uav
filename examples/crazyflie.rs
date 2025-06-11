use nalgebra::{vector, UnitQuaternion, Vector3};
use uav::control::QuadcopterController;
use uav::dynamics::{simulate_drone, Consts, State};

// Example usage
fn main() {
    let mut uav_state = State {
        position_x: 0.0,
        position_y: 0.0,
        position_z: 0.0,
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
        mass: 0.027,
        ixx: 1.4e-5,
        iyy: 1.4e-5,
        izz: 2.17e-5,
    };

    let simulation_span = 50.0; // 50 seconds
    let dt = 1. / 60.; // 60 fps
    let mut current_time = 0.0;

    let mut controller = QuadcopterController::new(
        dt,
        consts.mass,
        consts.ixx,
        consts.iyy,
        consts.izz,
        0.7,
        12.,
        5.,
        5.,
        0.03,
        0.26477955 / 4. * 0.7,
        0.26477955 / 4. * 1.5,
    );
    controller.set_gains(
        Vector3::new(25., 25., 5.),
        2.,
        6.,
        12.,
        2.,
        8.,
        1.,
        8.0,
        1.0,
    );
    while current_time < simulation_span {
        let t_pos = Vector3::new(-5., -5., 10.);
        let t_vel = Vector3::new(0.0, 0.0, 0.0);
        let t_acc = Vector3::new(0.0, 0.0, 0.0);
        let t_att = UnitQuaternion::identity();

        let est_pos = Vector3::new(
            uav_state.position_x,
            uav_state.position_y,
            uav_state.position_z,
        );
        let est_vel = Vector3::new(
            uav_state.velocity_x,
            uav_state.velocity_y,
            uav_state.velocity_z,
        );
        let est_omega = Vector3::new(
            -uav_state.roll_rate,
            -uav_state.pitch_rate,
            uav_state.yaw_rate,
        );
        let est_att =
            UnitQuaternion::from_euler_angles(uav_state.roll, uav_state.pitch, uav_state.yaw);

        let thrusts = controller.run_control(
            t_pos, t_vel, t_acc, t_att, est_pos, est_vel, est_omega, est_att,
        );

        let (forces, torques) = drone_dynamics([thrusts[0], thrusts[2], thrusts[3], thrusts[1]]);

        match simulate_drone(uav_state, consts, forces, torques, (0.0, dt), 1e-6) {
            Ok(new_state) => {
                uav_state = new_state.clone();

                println!(
                    "{} {} {}",
                    new_state.position_x, new_state.position_y, new_state.position_z
                );
            }
            Err(e) => println!("Simulation failed: {}", e),
        }

        current_time += dt;
    }
}

fn drone_dynamics(thrusts: [f64; 4]) -> (Vector3<f64>, Vector3<f64>) {
    let torque_to_thrust_ratio = 7.94e-12 / 3.16e-10;

    let rotor_1_position = vector![0.028, -0.028, 0.0];
    let rotor_2_position = vector![-0.028, -0.028, 0.0];
    let rotor_3_position = vector![-0.028, 0.028, 0.0];
    let rotor_4_position = vector![0.028, 0.028, 0.0];

    let f = vector![0.0, 0.0, 1.0];
    let f1 = f * thrusts[0];
    let f2 = f * thrusts[1];
    let f3 = f * thrusts[2];
    let f4 = f * thrusts[3];

    let full_force = f1 + f2 + f3 + f4;

    let t1_thrust = (rotor_1_position).cross(&(f1));
    let t1_torque = torque_to_thrust_ratio * (f1);

    let t2_thrust = (rotor_2_position).cross(&(f2));
    let t2_torque = torque_to_thrust_ratio * (f2);

    let t3_thrust = (rotor_3_position).cross(&(f3));
    let t3_torque = torque_to_thrust_ratio * (f3);

    let t4_thrust = (rotor_4_position).cross(&(f4));
    let t4_torque = torque_to_thrust_ratio * (f4);

    let t_thrust = t1_thrust + t2_thrust + t3_thrust + t4_thrust;
    let t_torque = (t1_torque - t4_torque) - (t2_torque - t3_torque);

    let torque = t_thrust - t_torque;

    return (
        Vector3::new(full_force.x, full_force.y, full_force.z),
        Vector3::new(torque.x, torque.y, torque.z),
    );
}
