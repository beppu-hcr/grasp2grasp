"""Write a minimal <object>.urdf (plus a symlink to <object>.obj) into out_dir for every mesh in mesh_dir,
so Isaac Gym can load the objects.
Mass/inertia are recomputed by Isaac Gym (override_com / override_inertia + density in the yaml).
Run from the repo root:
    python grasp_test/make_object_urdfs.py --mesh_dir data/mgg_pc/objects/obj --out_dir grasp_test/data/mgg_pc/objects/obj
"""
import os
import glob
import argparse

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{name}.obj" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{name}.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', default='data/mgg_pc/objects/obj')
    parser.add_argument('--out_dir', default='grasp_test/data/mgg_pc/objects/obj')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for obj_path in sorted(glob.glob(os.path.join(args.mesh_dir, '*.obj'))):
        name = os.path.splitext(os.path.basename(obj_path))[0]
        link_path = os.path.join(args.out_dir, f'{name}.obj')
        if not os.path.lexists(link_path):
            os.symlink(os.path.abspath(obj_path), link_path)
        urdf_path = os.path.join(args.out_dir, f'{name}.urdf')
        if os.path.exists(urdf_path) and not args.overwrite:
            continue
        with open(urdf_path, 'w') as f:
            f.write(URDF_TEMPLATE.format(name=name))
        n += 1
    print(f'Wrote {n} URDFs to {args.out_dir}')


if __name__ == '__main__':
    main()
