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
            {'xyz': [0.05355, -0.0725, 0.41492], 'rpy': [-0.9795, -0.5682, -2.3155], 'actuated': True}, # joints 1
            {'xyz': [0.03, 0.0, 0.1], 'rpy': [1.57079632679, 0.0, 0.0], 'actuated': True}, # joints 2
            {'xyz': [-0.03, 0.17283, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True}, # joints 7
            {'xyz': [-0.04188, 0.0, 0.07873], 'rpy': [1.57079632679, -1.57079632679, 0.0], 'actuated': True}, # joints 3
            {'xyz': [0.0405, 0.16461, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True}, # joints 4
            {'xyz': [-0.027, 0.0, 0.10039], 'rpy': [1.57079632679, 0.0, 0.0], 'actuated': True}, # joints 5
            {'xyz': [0.027, 0.029, 0.0], 'rpy': [-1.57079632679, 0.0, 0.0], 'actuated': True} # joints 6
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
    
    def manual_fk(self, joint_angles):
        """Manual Forward Kinematics"""
        angles = [0.0, 0.0] + list(joint_angles)
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
        
        return T[0:3, 3]
    
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
    
    def create_marker(self, position):
        """Create visual marker"""
        ball = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1])
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball, basePosition=position)

def main():
    """Random Motion Testing"""
    fk = YuMiFK()
    
    print("🎲 YuMi Random Motion FK Testing")
    print("Press Ctrl+C to stop")
    
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    fk.robot_id = p.loadURDF("/root/Downloads/yumi/yumi.urdf", useFixedBase=True)
    
    test_count = 0
    match_count = 0
    markers = []
    
    try:
        while True:
            test_count += 1
            
            # Generate random angles
            angles = [np.random.uniform(low, high) for low, high in fk.joint_limits]
            
            # Manual FK vs PyBullet FK
            manual_pos = fk.manual_fk(angles)
            pb_pos = fk.pybullet_fk(angles)
            
            diff = np.linalg.norm(manual_pos - pb_pos)
            if diff < 0.001:
                match_count += 1
            
            print(f"Test {test_count:3d} | Manual: [{manual_pos[0]:.4f}, {manual_pos[1]:.4f}, {manual_pos[2]:.4f}] | "
                  f"PyBullet: [{pb_pos[0]:.4f}, {pb_pos[1]:.4f}, {pb_pos[2]:.4f}] | "
                  f"Diff: {diff:.6f} | Accuracy: {100*match_count/test_count:.1f}%")
            
            # Create marker and clean old ones
            markers.append(fk.create_marker(manual_pos))
            if len(markers) > 10:
                p.removeBody(markers.pop(0))
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\nFinal: {match_count}/{test_count} ({100*match_count/test_count:.1f}%) matches")
    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
