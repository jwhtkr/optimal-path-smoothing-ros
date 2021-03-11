"""ROS node to expose the path smoothing MPC for use."""

import rospy

from nav_msgs.msg import Path

from path_smoothing.msg import FlatState, Trajectory
import path_smoothing.mpc as mpc
import tools.multi_array as multi_array


def path_to_traj(path):
    start_time = path.header.stamp.to_sec()
    times = [pose_stamped.header.stamp.to_sec() for pose_stamped in path.poses]
    poses = [pose_stamped.pose for pose_stamped in path.poses]

class DiffFlatMPCNode(mpc.DiffFlatMpc):
    """
    ROS node to expose the path smoothing MPC for use in ROS.

    Attributes
    ----------

    Parameters
    ----------

    """
    def __init__(self):  # noqa: D107
        rospy.init_node("diff_flat_mpc_node")
        parameters = self.get_parameters()
        super().__init__(*parameters)

        rospy.Subscriber("global_path", Path,
                         callback=self.global_path_cb, queue_size=10)
        # TODO: Add the iris and time-varying constraints
        # rospy.Subscriber("iris_regions", None, callback=self.iris_regions_cb,
        #                  queue_size=10)
        # rospy.Subscriber("time_varying_constraints", None,
        #                  callback=self.tv_constrs_cb, queue_size=10)
        rospy.Subscriber("flat_state", FlatState, callback=self.flat_state_cb,
                         queue_size=10)
        self.mpc_traj_pub = rospy.Publisher("mpc_traj", Trajectory,
                                            queue_size=10)

        self.global_path = None
        self.new_global_path = False
        self.iris_regions = None
        self.tv_constrs = None
        self.flat_state = None
        self.curr_control = None

    def global_path_cb(self, msg):
        self.global_path = msg
        self.new_global_path = True

    def iris_regions(self, msg):
        pass

    def tv_constrs_cb(self, msg):
        pass

    def flat_state_cb(self, msg):
        self.flat_state = multi_array.multi_array_to_array(msg.flat_state)
        self.flat_state = self.flat_state.reshape((msg.ndimensions,
                                                   msg.nderivatives), order="F")

    def run(self):
        if self.global_path:
            time = rospy.Time.now().to_sec()
            self.curr_desired_path = self.get_desired_traj(time)
            if self.new_global_path:
                self.get_iris_regions(self.global_path)
                self.get_tv_constrs(self.global_path)
                self.new_global_path = False
            self.mpc_output = self.mpc.step(time, self.flat_state,
                                            self.curr_control,
                                            self.curr_desired_traj)
            self.curr_control = self.mpc_output[1][:,0]
            self.mpc_traj_pub(self.mpc_to_traj(self.mpc_output))
