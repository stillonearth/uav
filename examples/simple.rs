use uav::dynamics::{simulate_drone, Consts, Forces, State, Torques};

// Example usage
fn main() {
    let initial_state = State {
        position_x: 0.0,
        position_y: 0.0,
        position_z: 1.0,
        velocity_x: 0.0,
        velocity_y: 0.0,
        velocity_z: 0.0,
        roll: 0.1, // Small initial roll
        pitch: 0.0,
        yaw: 0.0,
        roll_rate: 0.0,
        pitch_rate: 0.0,
        yaw_rate: 0.0,
    };

    let consts = Consts {
        g: 9.81,
        mass: 2.0, // 2 kg drone
        ixx: 0.1,
        iyy: 0.1,
        izz: 0.2,
    };

    let forces = Forces {
        x: 0.0,
        y: 0.0,
        z: consts.mass * consts.g, // Hover thrust
    };

    let torques = Torques {
        x: -0.1, // Small corrective roll torque
        y: 0.0,
        z: 0.0,
    };

    match simulate_drone(initial_state, consts, forces, torques, (0.0, 2.0), 1e-6) {
        Ok(final_state) => {
            println!("Final state:");
            println!(
                "Position: ({:.3}, {:.3}, {:.3})",
                final_state.position_x, final_state.position_y, final_state.position_z
            );
            println!(
                "Velocity: ({:.3}, {:.3}, {:.3})",
                final_state.velocity_x, final_state.velocity_y, final_state.velocity_z
            );
            println!(
                "Attitude: ({:.3}, {:.3}, {:.3})",
                final_state.roll, final_state.pitch, final_state.yaw
            );
            println!(
                "Rates: ({:.3}, {:.3}, {:.3})",
                final_state.roll_rate, final_state.pitch_rate, final_state.yaw_rate
            );
        }
        Err(e) => println!("Simulation failed: {}", e),
    }
}
