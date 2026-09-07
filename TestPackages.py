# debug_trimesh.py
import trimesh
import numpy as np

# Create a simple test mesh
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

print("Trimesh version:", trimesh.__version__)
print("\nAvailable attributes and methods:")
print(dir(mesh))

print("\nChecking specific attributes:")
print("'face_normals' in dir(mesh):", 'face_normals' in dir(mesh))
print("'update_normals' in dir(mesh):", 'update_normals' in dir(mesh))
print("'compute_face_normals' in dir(mesh):", 'compute_face_normals' in dir(mesh))

print("\nCurrent face_normals:", mesh.face_normals)
print("Is face_normals None?", mesh.face_normals is None)

# Try to access face_normals to trigger computation
try:
    _ = mesh.face_normals
    print("Accessed face_normals successfully.")
    print("face_normals after access:", mesh.face_normals)
except Exception as e:
    print("Error when accessing face_normals:", e)