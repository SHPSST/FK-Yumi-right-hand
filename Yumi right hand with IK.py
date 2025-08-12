import numpy as np
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description
from typing import Tuple, Optional, List, Dict, Any


class Yumi:
    """
    YuMi robot kinematics and inverse kinematics solver with full 6DOF control
    """
    
    def __init__(self):
        self.robot_id = None
        self.joint_data = [
            # Base transforms
            {'xyz': [0.0, 0.0, 0.1], 'rpy': [0.0, 0.0, 0.0], 'actuated': False},
            {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0], 'actuated': False},
            # Arm joints
            {'xyz': [0.05355, -0.0725, 0.41492], 'rpy': [-0.9795, -0.5682, -2.3155], 'actuated': True}, # joint 1
            {'xyz': [0.03, 0.0, 0.1], 'rpy': [1.57079632679, 0.0, 0.0], 'actuated': True}, # joint 2
            {'xyz': [-0.03, 0.17283, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True}, # joint 7
            {'xyz': [-0.04188, 0.0, 0.07873], 'rpy': [1.57079632679, -1.57079632679, 0.0], 'actuated': True}, # joint 3
            {'xyz': [0.0405, 0.16461, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True}, # joint 4
            {'xyz': [-0.027, 0.0, 0.10039], 'rpy': [1.57079632679, 0.0, 0.0], 'actuated': True}, # joint 5
            {'xyz': [0.027, 0.029, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True} # joint 6
        ]
        self.joint_limits = [[-2.941, 2.941], [-2.505, 0.759], [-2.941, 2.941], 
                            [-2.155, 1.396], [-5.061, 5.061], [-1.536, 2.409], [-3.997, 3.997]]
        
        # Workspace Constraints
        self.workspace_limits = {
            'min_reach': 0.2000, 'max_reach': 1.0068,
            'min_x': -0.3806, 'max_x': 0.6147,
            'min_y': -0.6210, 'max_y': 0.4083,
            'min_height': 0.2000, 'max_height': 0.9817,
            'max_side_reach': 0.5865,
        }

    # =============================================================================
    # WORKSPACE VALIDATION
    # =============================================================================
    
    def is_within_workspace(self, target_position: np.ndarray) -> Tuple[bool, Dict[str, bool]]:
        """ตรวจสอบว่าตำแหน่งเป้าหมายอยู่ใน workspace ที่ปลอดภัยหรือไม่"""
        x, y, z = target_position
        limits = self.workspace_limits
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        checks = {
            'reach_range': limits['min_reach'] <= distance <= limits['max_reach'],
            'x_range': limits['min_x'] <= x <= limits['max_x'],
            'y_range': limits['min_y'] <= y <= limits['max_y'],
            'z_range': limits['min_height'] <= z <= limits['max_height']
        }
        return all(checks.values()), checks
    
    def sample_safe_position(self) -> np.ndarray:
        """สุ่มตำแหน่งที่ปลอดภัยใน workspace"""
        limits = self.workspace_limits
        margin = 0.9
        
        x = np.random.uniform(limits['min_x'] * margin, limits['max_x'] * margin)
        y = np.random.uniform(limits['min_y'] * margin, limits['max_y'] * margin)
        z = np.random.uniform(limits['min_height'] * 1.1, limits['max_height'] * margin)
        
        distance = np.sqrt(x**2 + y**2 + z**2)
        if limits['min_reach'] * 1.1 <= distance <= limits['max_reach'] * margin:
            return np.array([x, y, z])
        else:
            return self.sample_safe_position()
    
    def sample_safe_orientation(self) -> np.ndarray:
        """สุ่ม orientation ที่สมเหตุสมผล (RPY)"""
        return np.array([
            np.random.uniform(-np.pi/3, np.pi/3),    # roll ±60°
            np.random.uniform(-np.pi/4, np.pi/4),    # pitch ±45°  
            np.random.uniform(-np.pi/2, np.pi/2)     # yaw ±90°
        ])

    # =============================================================================
    # ROTATION UTILITIES
    # =============================================================================
    
    def rpy_to_matrix(self, r: float, p: float, y: float) -> np.ndarray:
        """RPY to rotation matrix"""
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    def matrix_to_rpy(self, R: np.ndarray) -> np.ndarray:
        """Rotation matrix to RPY"""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])
    
    def axis_rotation(self, axis: List[float], angle: float) -> np.ndarray:
        """Axis-angle rotation"""
        if abs(angle) < 1e-10:
            return np.eye(3)
        axis = np.array(axis) / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # =============================================================================
    # FORWARD KINEMATICS
    # =============================================================================
    
    def forward_kinematics_with_transforms(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward kinematics with all transformation matrices"""
        angles = [0.0, 0.0] + list(joint_angles)
        transforms = []
        T = np.eye(4)
        
        for joint, angle in zip(self.joint_data, angles):
            R_base = self.rpy_to_matrix(joint['rpy'][0], joint['rpy'][1], joint['rpy'][2])
            if joint['actuated'] and abs(angle) > 1e-10:
                R_joint = self.axis_rotation([0, 0, 1], angle)
                R_total = R_base @ R_joint
            else:
                R_total = R_base
            
            T_joint = np.eye(4)
            T_joint[0:3, 0:3] = R_total
            T_joint[0:3, 3] = joint['xyz']
            T = T @ T_joint
            transforms.append(T.copy())
        
        return T, transforms
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward Kinematics - returns position and orientation (6DOF)"""
        T, _ = self.forward_kinematics_with_transforms(joint_angles)
        position = T[0:3, 3]
        orientation = self.matrix_to_rpy(T[0:3, 0:3])
        return position, orientation

    # =============================================================================
    # JACOBIAN COMPUTATION
    # =============================================================================
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute 6x7 Jacobian matrix for full 6DOF control"""
        T_final, transforms = self.forward_kinematics_with_transforms(joint_angles)
        p_end = T_final[0:3, 3]
        J = np.zeros((6, 7))
        
        actuated_count = 0
        for i, joint in enumerate(self.joint_data):
            if joint['actuated']:
                T_curr = transforms[i]
                z_i = T_curr[0:3, 2]  # Z-axis
                p_i = T_curr[0:3, 3]  # Position
                p_diff = p_end - p_i
                
                J[0:3, actuated_count] = np.cross(z_i, p_diff)  # Linear velocity
                J[3:6, actuated_count] = z_i  # Angular velocity
                
                actuated_count += 1
                if actuated_count >= 7:
                    break
        return J

    # =============================================================================
    # INVERSE KINEMATICS (6DOF)
    # =============================================================================
    
    def compute_orientation_error(self, current_rpy: np.ndarray, target_rpy: np.ndarray) -> np.ndarray:
        """คำนวณ orientation error ใน axis-angle form"""
        R_current = self.rpy_to_matrix(*current_rpy)
        R_target = self.rpy_to_matrix(*target_rpy)
        R_error = R_target @ R_current.T
        
        trace = np.trace(R_error)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if abs(angle) < 1e-6:
            return np.zeros(3)
        
        axis = np.array([R_error[2,1] - R_error[1,2], 
                        R_error[0,2] - R_error[2,0], 
                        R_error[1,0] - R_error[0,1]]) / (2 * np.sin(angle))
        
        return angle * axis
    
    def inverse_kinematics_step(self, current_angles: np.ndarray, target_position: np.ndarray, 
                               target_orientation: np.ndarray, position_weight: float = 1.0, 
                               orientation_weight: float = 0.5, ik_weight: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single step of Resolved Motion Rate Control for 6DOF Inverse Kinematics"""
        current_position, current_orientation = self.forward_kinematics(current_angles)
        position_error = np.array(target_position) - current_position
        orientation_error = self.compute_orientation_error(current_orientation, target_orientation)
        
        # Combined 6DOF error and Jacobian
        combined_error = np.concatenate([
            position_weight * position_error,
            orientation_weight * orientation_error
        ])
        J = self.jacobian(current_angles)
        J_weighted = np.vstack([
            position_weight * J[0:3, :],
            orientation_weight * J[3:6, :]
        ])
        
        # Pseudo-inverse with damping
        lambda_damping = 1e-4
        JJT = J_weighted @ J_weighted.T + lambda_damping * np.eye(J_weighted.shape[0])
        J_pinv = J_weighted.T @ np.linalg.inv(JJT)
        
        # Update joint angles with limits
        delta_q = ik_weight * (J_pinv @ combined_error)
        new_angles = current_angles + delta_q
        
        for i in range(len(new_angles)):
            if i < len(self.joint_limits):
                new_angles[i] = np.clip(new_angles[i], self.joint_limits[i][0], self.joint_limits[i][1])
        
        return new_angles, position_error, orientation_error
    
    def inverse_kinematics(self, target_position: np.ndarray, target_orientation: np.ndarray, 
                          initial_angles: Optional[np.ndarray] = None, max_iterations: int = 1000, 
                          position_tolerance: float = 1e-4, orientation_tolerance: float = 1e-3,
                          position_weight: float = 1.0, orientation_weight: float = 0.5, 
                          ik_weight: float = 0.1) -> Tuple[Optional[np.ndarray], bool, int]:
        """6DOF Inverse Kinematics solver (position + orientation required)"""
        # Workspace validation
        is_safe, checks = self.is_within_workspace(target_position)
        if not is_safe:
            return None, False, 0
        
        current_angles = (np.array(initial_angles) if initial_angles is not None 
                         else np.array([(low + high) / 2 for low, high in self.joint_limits]))
        
        adaptive_weight = ik_weight
        
        for iteration in range(max_iterations):
            new_angles, position_error, orientation_error = self.inverse_kinematics_step(
                current_angles, target_position, target_orientation, 
                position_weight, orientation_weight, adaptive_weight
            )
            
            # Check convergence for both position and orientation
            position_converged = np.linalg.norm(position_error) < position_tolerance
            orientation_converged = np.linalg.norm(orientation_error) < orientation_tolerance
            
            if position_converged and orientation_converged:
                return new_angles, True, iteration
            
            # Adaptive weight adjustment
            if iteration > 50:
                adaptive_weight = ik_weight * 0.8
            if iteration > 200:
                adaptive_weight = ik_weight * 0.5
                
            current_angles = new_angles
        
        return current_angles, False, max_iterations


# =============================================================================
# PYBULLET INTEGRATION
# =============================================================================

def get_pybullet_end_effector_pose(robot_id: int, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get end-effector pose from PyBullet simulation"""
    right_arm_joint_indices = [2, 3, 4, 5, 6, 7, 8] 

    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, right_arm_joint_indices[i], angle)
    
    end_effector_state = p.getLinkState(robot_id, 8)
    position = np.array(end_effector_state[0])
    orientation_rpy = np.array(p.getEulerFromQuaternion(end_effector_state[1]))
    
    return position, orientation_rpy


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def print_pose_comparison(title: str, target_pos: np.ndarray, target_ori: np.ndarray, 
                         custom_result: Tuple[np.ndarray, np.ndarray], 
                         pybullet_result: Tuple[np.ndarray, np.ndarray], 
                         solution: Optional[np.ndarray], converged: bool, iterations: int) -> None:
    """ฟังก์ชันสำหรับพิมพ์ผลการเปรียบเทียบ 6DOF"""
    print(f"🔹 {title}")
    print(f"Target Position:    [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
    print(f"Target Orientation: [{target_ori[0]:.6f}, {target_ori[1]:.6f}, {target_ori[2]:.6f}]")
    
    if converged and solution is not None:
        custom_pos, custom_ori = custom_result
        pybullet_pos, pybullet_ori = pybullet_result
        
        pos_error_custom = np.linalg.norm(custom_pos - target_pos)
        pos_error_pybullet = np.linalg.norm(pybullet_pos - target_pos)
        
        # ใช้ axis-angle error สำหรับ orientation (เหมือนใน IK)
        fk_instance = Yumi()  # สร้าง instance ชั่วคราว
        ori_error_custom = np.linalg.norm(fk_instance.compute_orientation_error(custom_ori, target_ori))
        ori_error_pybullet = np.linalg.norm(fk_instance.compute_orientation_error(pybullet_ori, target_ori))
        
        print(f"✅ 6DOF IK Converged in {iterations} iterations")
        print(f"Custom FK:")
        print(f"  Position:    [{custom_pos[0]:.6f}, {custom_pos[1]:.6f}, {custom_pos[2]:.6f}] Error: {pos_error_custom*1000:.3f} mm")
        print(f"  Orientation: [{custom_ori[0]:.6f}, {custom_ori[1]:.6f}, {custom_ori[2]:.6f}] Error: {np.degrees(ori_error_custom):.3f}°")
        
        print(f"PyBullet FK:")
        print(f"  Position:    [{pybullet_pos[0]:.6f}, {pybullet_pos[1]:.6f}, {pybullet_pos[2]:.6f}] Error: {pos_error_pybullet*1000:.3f} mm")
        print(f"  Orientation: [{pybullet_ori[0]:.6f}, {pybullet_ori[1]:.6f}, {pybullet_ori[2]:.6f}] Error: {np.degrees(ori_error_pybullet):.3f}°")
        
        print(f"Joint angles:    {[round(float(a), 3) for a in solution]}")
    else:
        print(f"❌ 6DOF IK FAILED - {'Outside workspace' if solution is None else f'Max iterations ({iterations})'}")
        
        if solution is not None and custom_result is not None:
            custom_pos, custom_ori = custom_result
            pos_error = np.linalg.norm(custom_pos - target_pos)
            
            # ใช้ axis-angle error สำหรับ orientation
            fk_instance = Yumi()
            ori_error = np.linalg.norm(fk_instance.compute_orientation_error(custom_ori, target_ori))
            
            print(f"   Achieved Position Error: {pos_error*1000:.3f} mm (Expected: <{1e-3*1000:.1f} mm)")
            print(f"   Achieved Orientation Error: {np.degrees(ori_error):.3f}° (Expected: <{np.degrees(1e-2):.1f}°)")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Test 6DOF IK (position + orientation control)"""
    fk = Yumi()

    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fk.robot_id = load_robot_description("yumi_description")

    initial_guess = [0.0, -0.5, 0.0, 0.5, 0.0, 0.5, 0.0]

    print("🔁 Testing 6DOF IK (Position + Orientation Control)...\n")

    # Tests 1-6: Full 6DOF control
    for test_num in range(1, 7):
        print("="*80)
        target_pos = fk.sample_safe_position()
        target_ori = fk.sample_safe_orientation()
        
        solution, converged, iterations = fk.inverse_kinematics(
            target_position=target_pos, target_orientation=target_ori,
            initial_angles=initial_guess, ik_weight=0.2,
            position_weight=1.0, orientation_weight=0.5,
            position_tolerance=1e-3, orientation_tolerance=1e-2, max_iterations=800
        )
        
        if solution is not None:
            custom_result = fk.forward_kinematics(solution)
            pybullet_result = get_pybullet_end_effector_pose(fk.robot_id, solution)
        else:
            custom_result = pybullet_result = (None, None)
        
        print_pose_comparison(f"Test {test_num}: 6DOF Control", target_pos, target_ori, 
                             custom_result, pybullet_result, solution, converged, iterations)
        print("="*80 + "\n")

    print("✅ All 6DOF tests completed!")
    input("Press Enter to exit...")
    p.disconnect()

if __name__ == "__main__":
    main()
