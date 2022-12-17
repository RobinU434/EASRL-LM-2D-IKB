import numpy as np


class RobotArm:
    def __init__(self, n_joints: int = 1, segment_lenght: float = 1) -> None:
        self._n_joints = n_joints
        self._segemnt_length = segment_lenght

        self._arm_length = self._n_joints * segment_lenght

        self._angles = np.zeros((self._n_joints))
        self._postions = np.zeros((self._n_joints + 1, 2))  # 2D, the plus one dim is the origin
        # init _postions
        self.set(self._angles)


    def reset(self):
        self._angles = np.zeros((self._n_joints))

    def set(self, angles):
        """applies given action to the arm 

        Args:
            angles (np.array): array with same length as number of joints
        """
        self._angles = angles % (2 * np.pi)
        for idx in range(self._n_joints):
            origin = self._postions[idx]
            
            # enw postion
            new_pos = np.array([np.cos(self._angles[idx]), np.sin(self._angles[idx])])
            new_pos *= self._segemnt_length

            # translate position
            new_pos += origin

            self._postions[idx + 1] = new_pos

    @property
    def angles(self):
        return self._angles

    @property
    def n_joints(self):
        return self._n_joints
    
    @property
    def arm_length(self):
        return self._arm_length

    @property
    def end_postion(self):
        return self._postions[-1]


if __name__ == "__main__":
    arm = RobotArm(20, 1)

    print(arm._postions)
    angles = np.zeros
    arm.set([0, np.pi / 2])
    print(arm._postions)
