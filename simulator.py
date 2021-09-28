import os

import gym
import time
import numpy as np

from world import World
import car
import dynamics
import visualize
import lane



class Simulation(object):
    def __init__(self, name, total_time=1000, recording_time=[0,1000]):
        self.name = name.lower()
        self.total_time = total_time
        self.recording_time = [max(0,recording_time[0]), min(total_time,recording_time[1])]
        self.frame_delay_ms = 0

    def reset(self):
        self.trajectory = []
        self.alreadyRun = False
        self.ctrl_array = [[0]*self.input_size]*self.total_time

    @property
    def ctrl(self):
        return self.ctrl_array 
    @ctrl.setter
    def ctrl(self, value):
        self.reset()
        self.ctrl_array = value.copy()
        self.run(reset=False)



class DrivingSimulation(Simulation):
    def __init__(self, name, total_time=50, recording_time=[0,50]):
        super(DrivingSimulation, self).__init__(name, total_time=total_time, recording_time=recording_time)
        self.world = World()
        clane = lane.StraightLane([0., -1.], [0., 1.], 0.17)
        self.world.lanes += [clane, clane.shifted(1), clane.shifted(-1)]
        self.world.roads += [clane]
        self.world.fences += [clane.shifted(2), clane.shifted(-2)]
        self.dyn = dynamics.CarDynamics(0.1)
        self.robot = car.Car(self.dyn, [0., -0.3, np.pi/2., 0.4], color='orange')
        self.human = car.Car(self.dyn, [0.17, 0., np.pi/2., 0.41], color='white')
        self.world.cars.append(self.robot)
        self.world.cars.append(self.human)
        self.initial_state = [self.robot.x, self.human.x]
        self.input_size = 2
        self.reset()
        self.viewer = None

    def initialize_positions(self):
        self.robot_history_x = []
        self.robot_history_u = []
        self.human_history_x = []
        self.human_history_u = []
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]

    def reset(self):
        super(DrivingSimulation, self).reset()
        self.initialize_positions()

    def run(self, reset=False):
        if reset:
            self.reset()
        else:
            self.initialize_positions()
        for i in range(self.total_time):
            self.robot.u = self.ctrl_array[i]
            if i < self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]]
            elif i < 2*self.total_time//5:
                self.human.u = [1., self.initial_state[1][3]]
            elif i < 3*self.total_time//5:
                self.human.u = [-1., self.initial_state[1][3]]
            elif i < 4*self.total_time//5:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            else:
                self.human.u = [0, self.initial_state[1][3]*1.3]
            self.robot_history_x.append(self.robot.x)
            self.robot_history_u.append(self.robot.u)
            self.human_history_x.append(self.human.x)
            self.human_history_u.append(self.human.u)
            self.robot.move()
            self.human.move()
            self.trajectory.append([self.robot.x, self.human.x])
        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj[self.recording_time[0]:self.recording_time[1]]

    def watch(self, repeat_count=1):
        self.robot.x = self.initial_state[0]
        self.human.x = self.initial_state[1]
        if self.viewer is None:
            self.viewer = visualize.Visualizer(0.1, magnify=1.2)
            self.viewer.main_car = self.robot
            self.viewer.use_world(self.world)
            self.viewer.paused = True
        for _ in range(repeat_count):
            video = self.viewer.run_modified(history_x=[self.robot_history_x, self.human_history_x], history_u=[self.robot_history_u, self.human_history_u])
        self.viewer.window.close()
        self.viewer = None
        return video
        


class FetchSimulation(Simulation):
    def __init__(self, total_time=1, recording_time=[0,1]):
        super(FetchSimulation, self).__init__(name='Fetch', total_time=total_time, recording_time=recording_time)

    def reset(self):
        super(FetchSimulation, self).reset()

    def run(self, reset=False):
        pass

    def get_trajectory(self, all_info=True):
        pass

    def get_recording(self, all_info=True):
        pass

    def watch(self, repeat_count=4):
        pass

    def close(self): # run only when you dont need the simulation anymore
        pass

    @property
    def seed(self):
        return 0
        
    @seed.setter
    def seed(self, value=0):
        pass