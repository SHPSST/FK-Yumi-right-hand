import numpy as np
import pybullet as p
import pybullet_data

class Yumi:
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
        self.joint_limits = [[-2.941, 2.941], [-2.505, 0.759], [-2.941, 2.941], [-2.155, 1.396], 
                            [-5.061, 5.061], [-1.536, 2.409], [-3.997, 3.997]]
        
        # Workspace Constraints
        self.workspace_limits = {
            'min_reach': 0.100,      # ระยะใกล้สุด 10cm
            'max_reach': 1.059,      # ระยะไกลสุด 105.9cm
            'min_height': 0.100,     # ความสูงต่ำสุด 10cm
            'max_height': 1.038,     # ความสูงสูงสุด 103.8cm
            'min_x': -0.380,         # ขอบเขต X ขั้นต่ำ
            'max_x': 0.636,          # ขอบเขต X สูงสุด
            'min_y': -0.626,         # ขอบเขต Y ขั้นต่ำ
            'max_y': 0.425,          # ขอบเขต Y สูงสุด
            'max_side_reach': 0.626  # การเอื้อมข้างสูงสุด
        }
    
    def is_within_workspace(self, target_position):
        """ตรวจสอบว่าตำแหน่งเป้าหมายอยู่ใน workspace ที่ปลอดภัยหรือไม่"""
        x, y, z = target_position
        limits = self.workspace_limits
        
        # คำนวณระยะทางจากฐาน
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        # ตรวจสอบทุกข้อจำกัด
        checks = {
            'reach_range': limits['min_reach'] <= distance <= limits['max_reach'],
            'x_range': limits['min_x'] <= x <= limits['max_x'],
            'y_range': limits['min_y'] <= y <= limits['max_y'],
            'z_range': limits['min_height'] <= z <= limits['max_height']
        }
        
        return all(checks.values()), checks
    
    def sample_safe_position(self):
        """สุ่มตำแหน่งที่ปลอดภัยใน workspace"""
        limits = self.workspace_limits
        
        # สุ่มในพื้นที่ปลอดภัย (เก็บ margin 10%)
        margin = 0.9
        
        x = np.random.uniform(limits['min_x'] * margin, limits['max_x'] * margin)
        y = np.random.uniform(limits['min_y'] * margin, limits['max_y'] * margin)
        z = np.random.uniform(limits['min_height'] * 1.1, limits['max_height'] * margin)
        
        # ตรวจสอบว่าอยู่ในช่วง reach ที่ปลอดภัย
        distance = np.sqrt(x**2 + y**2 + z**2)
        if limits['min_reach'] * 1.1 <= distance <= limits['max_reach'] * margin:
            return np.array([x, y, z])
        else:
            # ถ้าไม่อยู่ในช่วง ให้สุ่มใหม่
            return self.sample_safe_position()
    
    def rpy_to_matrix(self, r, p, y):
        """RPY to rotation matrix"""
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    def axis_rotation(self, axis, angle):
        """Axis-angle rotation"""
        if abs(angle) < 1e-10:
            return np.eye(3)
        axis = np.array(axis) / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    def forward_kinematics_with_transforms(self, joint_angles):
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
    
    def forward_kinematics(self, joint_angles):
        """Forward Kinematics - returns end-effector position"""
        T, _ = self.forward_kinematics_with_transforms(joint_angles)
        return T[0:3, 3]
    
    def jacobian(self, joint_angles):
        """Compute Jacobian matrix"""
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
    
    def pseudo_inverse(self, joint_angles):
        """Compute pseudo-inverse of Jacobian"""
        J = self.jacobian(joint_angles)
        return np.linalg.pinv(J)
    
    def inverse_kinematics_step(self, current_angles, target_position, weight=0.1):
        """Single step of Resolved Motion Rate Control for Inverse Kinematics"""
        current_position = self.forward_kinematics(current_angles)
        position_error = np.array(target_position) - current_position

        # Use only position part of Jacobian (3x7)
        J = self.jacobian(current_angles)
        J_pos = J[0:3, :]
        
        # ปรับปรุง pseudo-inverse ด้วย damping
        lambda_damping = 1e-4
        JJT = J_pos @ J_pos.T + lambda_damping * np.eye(3)
        J_pinv = J_pos.T @ np.linalg.inv(JJT)
        
        # Calculate joint angle changes
        delta_q = weight * (J_pinv @ position_error)
        
        # Update joint angles
        new_angles = current_angles + delta_q
        
        # Apply joint limits
        for i in range(len(new_angles)):
            if i < len(self.joint_limits):
                new_angles[i] = np.clip(new_angles[i], self.joint_limits[i][0], self.joint_limits[i][1])
        
        return new_angles, position_error
    
    def inverse_kinematics(self, target_position, initial_angles=None, max_iterations=1000, tolerance=1e-4, weight=0.1):
        """Inverse Kinematics solver with workspace validation"""
        # ตรวจสอบ workspace ก่อน
        is_safe, checks = self.is_within_workspace(target_position)
        if not is_safe:
            return None, False, 0
        
        if initial_angles is None:
            current_angles = np.array([(low + high) / 2 for low, high in self.joint_limits])
        else:
            current_angles = np.array(initial_angles)
        
        adaptive_weight = weight
        
        for iteration in range(max_iterations):
            new_angles, position_error = self.inverse_kinematics_step(current_angles, target_position, adaptive_weight)
            error_magnitude = np.linalg.norm(position_error)
                
            if error_magnitude < tolerance:
                return new_angles, True, iteration
            
            if iteration > 50:
                adaptive_weight = weight * 0.8
            if iteration > 200:
                adaptive_weight = weight * 0.5
                
            current_angles = new_angles
        
        return current_angles, False, max_iterations

def get_pybullet_end_effector_position(robot_id, joint_angles):
    """Get end-effector position from PyBullet simulation"""
    # ตั้งค่า joint angles สำหรับแขนขวา (joints 2-8 ใน PyBullet)
    right_arm_joint_indices = [2, 3, 4, 5, 6, 7, 8]  # Right arm joints in PyBullet
    
    # ตั้งค่า joint positions
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, right_arm_joint_indices[i], angle)
    
    # Get end-effector link state (สมมติว่าเป็น link สุดท้ายของแขนขวา)
    # ใช้ link index 8 สำหรับ end-effector ของแขนขวา
    end_effector_state = p.getLinkState(robot_id, 8)
    position = end_effector_state[0]  # World position
    
    return np.array(position)

def main():
    """Test 10 IK cases with full detailed output (like original code)"""
    fk = Yumi()

    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fk.robot_id = p.loadURDF("/root/Downloads/yumi/yumi.urdf", useFixedBase=True)

    initial_guess = [0.0, -0.5, 0.0, 0.5, 0.0, 0.5, 0.0]

    print("🔁 Testing 10 random IK cases (full detailed output)...\n")

    for case_num in range(1, 11):
        print(f"🔹 Case {case_num}")

        # Generate safe target position
        safe_target = fk.sample_safe_position()

        # Solve IK
        solution, converged, iterations = fk.inverse_kinematics(
            target_position=safe_target,
            initial_angles=initial_guess,
            weight=0.3,
            tolerance=1e-3,
            max_iterations=500
        )

        print(f"IK Input:       [{safe_target[0]:.6f}, {safe_target[1]:.6f}, {safe_target[2]:.6f}]")

        if converged and solution is not None:
            # Verify FK with IK solution
            custom_fk_result = fk.forward_kinematics(solution)
            pybullet_fk_result = get_pybullet_end_effector_position(fk.robot_id, solution)

            # Calculate errors
            custom_fk_error = np.linalg.norm(custom_fk_result - safe_target)
            pybullet_fk_error = np.linalg.norm(pybullet_fk_result - safe_target)

            print(f"✅ IK Converged in {iterations} iterations")
            print(f"Custom FK:      [{custom_fk_result[0]:.6f}, {custom_fk_result[1]:.6f}, {custom_fk_result[2]:.6f}] Error: {custom_fk_error:.6f}")
            print(f"PyBullet FK:    [{pybullet_fk_result[0]:.6f}, {pybullet_fk_result[1]:.6f}, {pybullet_fk_result[2]:.6f}] Error: {pybullet_fk_error:.6f}")
            print(f"Joint angles:   {[round(float(a), 3) for a in solution]}")
        else:
            print(f"❌ IK FAILED - Reason: {'Outside workspace' if solution is None else f'Max iterations reached ({iterations})'}")

            if solution is not None:
                custom_fk_result = fk.forward_kinematics(solution)
                pybullet_fk_result = get_pybullet_end_effector_position(fk.robot_id, solution)
                custom_fk_error = np.linalg.norm(custom_fk_result - safe_target)
                pybullet_fk_error = np.linalg.norm(pybullet_fk_result - safe_target)

                print(f"Best attempt:")
                print(f"Custom FK:      [{custom_fk_result[0]:.6f}, {custom_fk_result[1]:.6f}, {custom_fk_result[2]:.6f}] Error: {custom_fk_error:.6f}")
                print(f"PyBullet FK:    [{pybullet_fk_result[0]:.6f}, {pybullet_fk_result[1]:.6f}, {pybullet_fk_result[2]:.6f}] Error: {pybullet_fk_error:.6f}")
                print(f"Joint angles:   {[round(float(a), 3) for a in solution]}")

        print("\n" + "-"*80 + "\n")

    input("✅ All 10 cases done. Press Enter to exit...")
    p.disconnect()

if __name__ == "__main__":
    main()
