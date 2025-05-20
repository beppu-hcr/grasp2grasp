# use "bash build_desc.sh -h" to see these instructions in the terminal


      BUILD_DESC.SH

   INSTRUCTIONS FOR USE
   --------------------

'build_desc.sh' can be called with up to two (2) arguments.
The first argument is the side of the hand, left or right, and
the second argument determines the use of a robot tree diagram.


Examples:

   $ bash build_desc.sh RIGHT 1
   Builds the right hand URDF from its xacro file and shows the robot tree diagram when finished.

   $ bash build_desc.sh left
   Builds the left hand URDF from its xacro file and DOES NOT show the tree diagram.



Possible variations of the 1st argument are:
 RIGHT, Right, right, R, r, LEFT, Left, left, L and l.

For the 2nd argument, '1' produces the tree graph.
Left blank or passed any other value, the graph will not be created

URDFPY ordering:
['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip',
'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_7.0_tip',
'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_11.0_tip',
'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_15.0_tip']

Data ordering:
['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip',
'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_7.0_tip',
'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 'joint_11.0_tip',
'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 'joint_15.0_tip']
