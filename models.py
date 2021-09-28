from simulator import DrivingSimulation, FetchSimulation
import numpy as np



class Driver(DrivingSimulation):
    """
    Original Driver model from 'Asking easy questions: A user-friendly approach to active reward learning'
    Bıyık, E., Palan, M., Landolfi, N. C., Losey, D. P., & Sadigh, D. (2019).arXiv preprint arXiv:1910.04365.
    """
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driver', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 4

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)
        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30 * np.min(
            [np.square(recording[:, 0, 0] - 0.17), np.square(recording[:, 0, 0]), np.square(recording[:, 0, 0] + 0.17)],
            axis=0))) / 0.15343634
        # keeping speed (lower is better)
        # keeping_speed = np.mean(np.square(recording[:, 0, 3] - 1)) / 0.42202643
        keeping_speed = np.mean(np.square(recording[:, 0, 3] - 1)) / 0.42202643
        # heading (higher is better)
        heading = np.mean(np.sin(recording[:, 0, 2])) / 0.06112367
        # collision avoidance (lower is better)
        collision_avoidance = np.mean(np.exp(-(7 * np.square(recording[:, 0, 0] - recording[:, 1, 0]) + 3 * np.square(
            recording[:, 0, 1] - recording[:, 1, 1])))) / 0.15258019
        return [staying_in_lane, keeping_speed, heading, collision_avoidance]

    @property
    def state(self):
        return [self.robot.x, self.human.x]
    @state.setter
    def state(self, value):
        self.reset()
        self.initial_state = value.copy()

    def set_ctrl(self, value):
        arr = [[0]*self.input_size]*self.total_time
        interval_count = len(value)//self.input_size
        interval_time = int(self.total_time / interval_count)
        arr = np.array(arr).astype(float)
        j = 0
        for i in range(interval_count):
            arr[i*interval_time:(i+1)*interval_time] = [value[j], value[j+1]]
            j += 2
        self.ctrl = list(arr)

    def feed(self, value):
        ctrl_value = value[:]
        self.set_ctrl(ctrl_value)

    def get_cost_given_input(self, input):
        """

        :param input:
        :param weight:
        :return:
        """
        self.feed(list(input))
        features = np.array(self.get_features())
        return -np.dot(self.weights, features)  # minus as we want to maximize

    def find_optimal_path(self, weights):
        """
        New function to numerically find an optimal trajectory given weights
        Note: Using a generic numerical solver can lead to suboptimal solutions.
        :param weights:
        :param lb_input:
        :param ub_input:
        :return: optimal_controls, path_features, path_cost
        """
        from scipy.optimize import minimize
        self.weights = weights[0:self.num_of_features]
        lb_input = [x[0] for x in self.feed_bounds]
        ub_input = [x[1] for x in self.feed_bounds]
        random_start = [0] * self.feed_size
        random_start = np.random.rand(self.feed_size)
        bounds = np.transpose([lb_input, ub_input])
        res = minimize(self.get_cost_given_input, x0=random_start, bounds=bounds, method='L-BFGS-B')
        self.feed(list(res.x))
        features = np.array(self.get_features())
        controls = res.x
        return controls, features, -res.fun

class DriverExtended(Driver):
    """
    Extended 10 dimensional driver model
    """
    def __init__(self, total_time=50, recording_time=[0,50]):
        super(Driver ,self).__init__(name='driverextended', total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 10
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.ctrl_bounds = [(-1,1)]*self.ctrl_size
        self.state_bounds = []
        self.feed_bounds = self.state_bounds + self.ctrl_bounds
        self.num_of_features = 10

    def get_features(self):
        recording = self.get_recording(all_info=False)
        recording = np.array(recording)
        # staying in lane (higher is better)
        staying_in_lane = np.mean(np.exp(-30*np.min([np.square(recording[:,0,0]-0.17), np.square(recording[:,0,0]), np.square(recording[:,0,0]+0.17)], axis=0))) / 0.15343634

        # keeping speed (lower is better)
        keeping_speed = -np.mean(np.square(recording[:,0,3]-1)) / 0.42202643

        # heading (higher is better)
        heading = np.mean(np.sin(recording[:,0,2])) / 0.06112367

        # collision avoidance (lower is better)
        collision_avoidance = -np.mean(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.15258019

        # min collision avoidance over time (lower is better)
        min_collision_avoidance = -np.max(np.exp(-(7*np.square(recording[:,0,0]-recording[:,1,0])+3*np.square(recording[:,0,1]-recording[:,1,1])))) / 0.10977646

        # average jerk (lower is better)
        acceleration = recording[1:,0,3] - recording[:-1,0,3]
        average_jerk = -np.mean(np.abs(acceleration[1:] - acceleration[:-1])) / 0.00317041

        # vertical displacement (higher is better)
        vertical_displacement = (recording[-1,0,1] - recording[0,0,1]) / 1.01818467


        final_left_lane = (recording[-1, 0, 0] > -.25) and (recording[-1, 0, 0] < -.09)
        final_right_lane = (recording[-1, 0, 0] > .09) and (recording[-1, 0, 0] < .25)
        final_center_lane = (recording[-1, 0, 0] > -.09) and (recording[-1, 0, 0] < .09)

        return [staying_in_lane,
                keeping_speed,
                heading,
                collision_avoidance,
                min_collision_avoidance,
                average_jerk,
                vertical_displacement,
                final_left_lane,
                final_right_lane,
                final_center_lane
                ]


class Fetch(FetchSimulation):
    def __init__(self, total_time=1, recording_time=[0,1]):
        super(Fetch ,self).__init__(total_time=total_time, recording_time=recording_time)
        self.ctrl_size = 1
        self.state_size = 0
        self.feed_size = self.ctrl_size + self.state_size
        self.num_of_features = 8

    def get_features(self):
        A = np.load('ctrl_samples/fetch.npz')
        return list(A['feature_set'][self.ctrl,:])

    @property
    def state(self):
        return 0
    @state.setter
    def state(self, value):
        pass

    def set_ctrl(self, value):
        self.ctrl = value

    def feed(self, value):
        self.set_ctrl(value)
