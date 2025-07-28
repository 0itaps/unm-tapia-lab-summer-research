import torch
import pytorch_kinematics as pk
import numpy as np
import polyscope as ps

urdf_path = "kuka_iiwa/model.urdf"
with open(urdf_path, 'r') as f:
    urdf_str = f.read()

chain = pk.build_chain_from_urdf(urdf_str)

n_joints = len(chain.get_joint_parameter_names())
q = torch.zeros((1, n_joints))

fk_results = chain.forward_kinematics(q)

positions = []
for link_name, transform in fk_results.items():
    matrix = transform.get_matrix().detach().numpy()[0]
    pos = matrix[:3, 3]
    positions.append(pos)

positions = np.array(positions)

ps.init()
ps.set_up_dir("z_up")
ps.register_point_cloud("Robot Links", positions)
ps.show()
