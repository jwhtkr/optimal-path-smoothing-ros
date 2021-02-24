"""ROS node to expose the path smoothing MPC for use."""

import rospy

from path_smoothing.msg import FlatState, Trajectory
import path_smoothing.mpc as mpc


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

        rospy.Subscriber("global_path", Trajectory,
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

    def global_path_cb(self, msg):
        pass

    def iris_regions(self, msg):
        pass

    def tv_constrs_cb(self, msg):
        pass

    def flat_state_cb(self, msg):
        pass

    def run(self):
        pass
