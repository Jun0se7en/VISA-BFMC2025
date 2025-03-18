import numpy as np

class KinematicBicycleModel:
    def __init__(self, length_rear: float, delta_time: float):
        self.delta_time = delta_time
        self.length_rear = length_rear

    def discrete_kbm(self,x: float, y: float, heading: float,velocity: float,angular_velocity: float, dt: float):

        if(velocity == 0):

            slip_angle = 0

        else:
            
            slip_angle = np.degrees(np.arcsin(np.radians((angular_velocity * self.length_rear) / velocity)))
            
        new_x = x + velocity * np.cos(np.radians(heading + slip_angle)) * dt

        new_y = y + velocity * np.sin(np.radians(heading + slip_angle)) * dt

        new_heading = heading + angular_velocity * dt
        
        return new_x, new_y, new_heading ,slip_angle
