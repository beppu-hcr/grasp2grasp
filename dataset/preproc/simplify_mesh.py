import open3d as o3d
import os
import argparse

def simplify_mesh(input_path, output_path, reduction_factor):
    # Read the mesh from file
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        print(f"Warning: Mesh {input_path} is empty or failed to load.")
        return

    # Ensure normals are computed for proper processing
    mesh.compute_vertex_normals()

    # Determine original triangle count and compute target
    original_triangles = len(mesh.triangles)
    target_triangles = int(original_triangles * reduction_factor)
    
    # Check if the target triangle count is reasonable
    if target_triangles < 4:
        print(f"Mesh {input_path} has too few triangles to simplify (original: {original_triangles}). Skipping.")
        return

    print(f"Simplifying {input_path}: {original_triangles} -> {target_triangles} triangles.")

    # Simplify the mesh using quadric decimation
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    simplified_mesh.compute_vertex_normals()

    # Save the simplified mesh
    o3d.io.write_triangle_mesh(output_path, simplified_mesh)
    print(f"Simplified mesh saved to {output_path}")

def process_folder(input_dir, output_dir, reduction_factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".stl"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            simplify_mesh(input_path, output_path, reduction_factor)

def main():
    parser = argparse.ArgumentParser(description="Simplify STL meshes using Open3D by reducing faces to a fraction of the original.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing STL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to folder to save simplified STL files")
    parser.add_argument("--reduction_factor", type=float, default=0.3333, 
                        help="Fraction of original triangles to keep (e.g., 0.3333 for 1/3 of the original faces)")
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir, args.reduction_factor)

if __name__ == "__main__":
    main()