# Void controller - does nothing, just keeps the robot in place
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

while robot.step(timestep) != -1:
    pass
