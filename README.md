# autonomous_vehicles_project
## Functionality implemented:
- controller_node included in the keyboard file, in which the publishing of
car control using WASD keys (W is for throttle, D is for brake,
AD for turning), changing gears (keys 1-4)
- the states_node, contained in the states file, realizes the publication of the turn value and
the vehicle speed converted from /base_pose_ground_truth topic to the topic
/prius/states
- visualizer_node, contained in the visualizer file, publishes the front camera image with
overlaid with information about the driving mode (collect, selfdriving), and the current
values of speed, throttle, brake, steer as text on the image, all of which is
published in the /prius/visualization topic
- collector_node included in the collector file, responsible for collecting data
for network training, saves a frame from the vehicle's front camera to a png file, and
the corresponding values of turn and speed at a given moment (saved to a
csv file).
- The control_prediction_node, contained in the control_prediction file, with the
selfdriving mode is enabled, it publishes the values of turn and speed predicted by the neural network model.
speed, the speed is converted by the PID controller into throttle
and brake values,

## Neural Network Model:
A convolutional neural network was used, with the architecture shown below:
