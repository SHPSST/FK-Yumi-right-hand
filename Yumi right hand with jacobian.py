import numpy as np
import pybullet as p
import pybullet_data
import time

class YuMiFK:
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
    
    def manual_fk(self, joint_angles):
        """Manual Forward Kinematics"""
        T, _ = self.forward_kinematics_with_transforms(joint_angles)
        return T[0:3, 3]
    
    def cross_product_jacobian(self, joint_angles):
        """Cross Product Jacobian Method"""
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
    
    def pybullet_fk(self, joint_angles):
        """PyBullet FK"""
        indices = [2, 3, 4, 5, 6, 7, 8]
        
        for i in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, i, 0.0)
        
        for idx, angle in zip(indices, joint_angles):
            p.resetJointState(self.robot_id, idx, angle)
        
        p.stepSimulation()
        link_state = p.getLinkState(self.robot_id, 8, computeForwardKinematics=True)
        return np.array(link_state[0])
    
    def pybullet_jacobian(self, joint_angles):
        """PyBullet Jacobian"""
        joint_indices = [2, 3, 4, 5, 6, 7, 8]
        
        for i in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, i, 0.0)
        
        for idx, angle in zip(joint_indices, joint_angles):
            p.resetJointState(self.robot_id, idx, angle)
        
        p.stepSimulation()
        
        all_movable_joints = []
        all_positions = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            if joint_type in [0, 1]:  # REVOLUTE or PRISMATIC
                all_movable_joints.append(i)
                joint_state = p.getJointState(self.robot_id, i)
                all_positions.append(joint_state[0])
        
        all_velocities = [0.0] * len(all_movable_joints)
        all_accelerations = [0.0] * len(all_movable_joints)
        
        jac_linear, jac_angular = p.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=8,
            localPosition=[0.0, 0.0, 0.0],
            objPositions=all_positions,
            objVelocities=all_velocities,
            objAccelerations=all_accelerations
        )
        
        J_full = np.vstack([np.array(jac_linear), np.array(jac_angular)])
        
        right_hand_cols = []
        for joint_idx in joint_indices:
            if joint_idx in all_movable_joints:
                col_idx = all_movable_joints.index(joint_idx)
                right_hand_cols.append(col_idx)
        
        return J_full[:, right_hand_cols]
    
    def compare_jacobians(self, joint_angles, test_num=1, verbose=False):
        """Compare Manual vs PyBullet Jacobians"""
        manual_pos = self.manual_fk(joint_angles)
        pb_pos = self.pybullet_fk(joint_angles)
        
        J_manual = self.cross_product_jacobian(joint_angles)
        J_pb = self.pybullet_jacobian(joint_angles)
        
        fk_diff = np.linalg.norm(manual_pos - pb_pos)
        jac_diff = np.linalg.norm(J_manual - J_pb)
        
        if verbose:
            print(f"\n📋 Test {test_num} Detailed Results:")
            print("=" * 80)
            
            # FK Comparison
            print("🎯 Forward Kinematics:")
            print(f"  Manual:   [{manual_pos[0]:8.5f}, {manual_pos[1]:8.5f}, {manual_pos[2]:8.5f}]")
            print(f"  PyBullet: [{pb_pos[0]:8.5f}, {pb_pos[1]:8.5f}, {pb_pos[2]:8.5f}]")
            print(f"  Difference: {fk_diff:.8f} {'✅' if fk_diff < 1e-6 else '❌'}")
            
            # Jacobian Comparison (show first 2 columns and last column)
            print("\n🔧 Jacobian Matrix (Linear Part - First 2 joints + Last joint):")
            print("     Method    |    J1_x       J1_y       J1_z   |    J2_x       J2_y       J2_z   |    J7_x       J7_y       J7_z")
            print("  -------------|------------------------------|------------------------------|------------------------------")
            print(f"  Manual       | {J_manual[0,0]:8.5f}  {J_manual[1,0]:8.5f}  {J_manual[2,0]:8.5f} | {J_manual[0,1]:8.5f}  {J_manual[1,1]:8.5f}  {J_manual[2,1]:8.5f} | {J_manual[0,6]:8.5f}  {J_manual[1,6]:8.5f}  {J_manual[2,6]:8.5f}")
            print(f"  PyBullet     | {J_pb[0,0]:8.5f}  {J_pb[1,0]:8.5f}  {J_pb[2,0]:8.5f} | {J_pb[0,1]:8.5f}  {J_pb[1,1]:8.5f}  {J_pb[2,1]:8.5f} | {J_pb[0,6]:8.5f}  {J_pb[1,6]:8.5f}  {J_pb[2,6]:8.5f}")
            
            print("\n🔧 Jacobian Matrix (Angular Part - First 2 joints + Last joint):")
            print("     Method    |    J1_x       J1_y       J1_z   |    J2_x       J2_y       J2_z   |    J7_x       J7_y       J7_z")
            print("  -------------|------------------------------|------------------------------|------------------------------")
            print(f"  Manual       | {J_manual[3,0]:8.5f}  {J_manual[4,0]:8.5f}  {J_manual[5,0]:8.5f} | {J_manual[3,1]:8.5f}  {J_manual[4,1]:8.5f}  {J_manual[5,1]:8.5f} | {J_manual[3,6]:8.5f}  {J_manual[4,6]:8.5f}  {J_manual[5,6]:8.5f}")
            print(f"  PyBullet     | {J_pb[3,0]:8.5f}  {J_pb[4,0]:8.5f}  {J_pb[5,0]:8.5f} | {J_pb[3,1]:8.5f}  {J_pb[4,1]:8.5f}  {J_pb[5,1]:8.5f} | {J_pb[3,6]:8.5f}  {J_pb[4,6]:8.5f}  {J_pb[5,6]:8.5f}")
            
            print(f"\n📊 Overall Jacobian Difference: {jac_diff:.8f} {'✅' if jac_diff < 0.01 else '❌'}")
            print("=" * 80)
        
        return {
            'fk_diff': fk_diff,
            'jacobian_diff': jac_diff,
            'manual_jacobian': J_manual,
            'pybullet_jacobian': J_pb,
            'manual_pos': manual_pos,
            'pb_pos': pb_pos
        }
    
    def create_marker(self, position, color=[1, 0, 0, 1]):
        """Create visual marker"""
        ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball, basePosition=position)

def main():
    """Test FK and Jacobian with 5 random detailed tests"""
    fk = YuMiFK()
    
    print("🤖 YuMi FK and Jacobian Testing - 5 Random Test Cases")
    
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fk.robot_id = p.loadURDF("/root/Downloads/yumi/yumi.urdf", useFixedBase=True)
    
    # Run 5 detailed tests with visual markers
    results = []
    markers = []
    
    for i in range(5):
        print(f"\n🧪 Test Case {i+1}/5")
        angles = [np.random.uniform(low, high) for low, high in fk.joint_limits]
        print(f"Joint Angles: [{', '.join([f'{a:.4f}' for a in angles])}]")
        
        result = fk.compare_jacobians(angles, test_num=i+1, verbose=True)
        results.append(result)
        
        # Check if FK positions match (within tolerance)
        fk_match = result['fk_diff'] < 1e-6
        
        if fk_match:
            # Single blue marker if positions match
            blue_marker = fk.create_marker(result['manual_pos'], [0, 0, 1, 1])  # Blue
            markers.append(blue_marker)
            print(f"🔵 Blue marker: FK positions match perfectly!")
        else:
            # Separate red and green markers if positions don't match
            red_marker = fk.create_marker(result['manual_pos'], [1, 0, 0, 1])    # Red for manual
            green_marker = fk.create_marker(result['pb_pos'], [0, 1, 0, 1])      # Green for PyBullet
            markers.extend([red_marker, green_marker])
            print(f"🔴 Red marker: Manual FK position")
            print(f"🟢 Green marker: PyBullet FK position")
            print(f"⚠️  Positions differ by {result['fk_diff']:.8f}")
        
        # Add a brief pause between tests
        time.sleep(2)
    
    # Summary of all 5 tests
    print("\n📊 Summary of All 5 Tests:")
    print("=" * 60)
    fk_matches = 0
    jac_matches = 0
    
    for i, result in enumerate(results):
        fk_ok = result['fk_diff'] < 1e-6
        jac_ok = result['jacobian_diff'] < 0.01
        
        if fk_ok: fk_matches += 1
        if jac_ok: jac_matches += 1
        
        print(f"Test {i+1}: FK Diff {result['fk_diff']:.8f} {'✅' if fk_ok else '❌'} | "
              f"Jac Diff {result['jacobian_diff']:.6f} {'✅' if jac_ok else '❌'}")
    
    print(f"\n🎯 Overall Results:")
    print(f"FK Perfect Matches: {fk_matches}/5 ({100*fk_matches/5:.1f}%)")
    print(f"Jacobian Good Matches: {jac_matches}/5 ({100*jac_matches/5:.1f}%)")
    
    fk_diffs = [r['fk_diff'] for r in results]
    jac_diffs = [r['jacobian_diff'] for r in results]
    print(f"FK Avg Error: {np.mean(fk_diffs):.8f}")
    print(f"Jacobian Avg Error: {np.mean(jac_diffs):.6f}")
    
    # Count different marker types
    blue_count = sum(1 for i, result in enumerate(results) if result['fk_diff'] < 1e-6)
    mismatch_count = len(results) - blue_count
    
    print(f"\n🎨 Visual Markers in Simulation:")
    print(f"🔵 Blue spheres: FK positions match perfectly ({blue_count} markers)")
    if mismatch_count > 0:
        print(f"🔴 Red spheres: Manual FK positions ({mismatch_count} markers)")
        print(f"🟢 Green spheres: PyBullet FK positions ({mismatch_count} markers)")
    print(f"📍 Total markers created: {len(markers)}")
    
    print(f"\n⏸️  Simulation is kept open for inspection.")
    print(f"🔍 Examine the markers in the 3D view to verify FK accuracy.")
    input("Press Enter to close simulation and exit...")
    
    p.disconnect()

if __name__ == "__main__":
    main()
