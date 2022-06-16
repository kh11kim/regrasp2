import numpy as np
from regrasp.utils.world import BulletWorld
import trimesh
from itertools import combinations
import matplotlib.pyplot as plt

def normal(triangle):
    edge1 = triangle[0] - triangle[1]
    edge2 = triangle[1] - triangle[2]
    vec = np.cross(edge1, edge2)
    return vec/np.linalg.norm(vec)

def angle_between_triangles(tri1, tri2):
    is_coplanar = trimesh.triangles.all_coplanar([tri1, tri2])
    normals = trimesh.triangles.normals([tri1, tri2])[0]
    normal1, normal2 = normals[0], normals[1]
    if normal1@normal2 < 0:
        normal2 = -normal2
    return np.arccos(np.dot(normal1, normal2))

def main():
    mesh = trimesh.load_mesh("data/urdfs/block/meshes/base_link.STL")
    #world = BulletWorld(gui=True)
    #world.load_urdf("block1", "data/urdfs/block/urdf/block.urdf")
    # tri1 = mesh.triangles[0]
    # tri2 = mesh.triangles[1]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    rng = [-0.017, 0.017]
    eps = 0.01
    triangles = mesh.triangles.copy()
    indices = list(range(len(mesh.triangles)))
    
    # triangle clustering
    clusters = []
    while len(indices) != 0:
        i = indices[0]
        cluster = []
        for j in indices:
            if trimesh.triangles.all_coplanar([triangles[i], triangles[j]]):
                cluster.append(j)
        clusters.append(tuple(cluster))
        indices = [k for k in indices if k not in cluster]

    #TODO

    for tri1, tri2 in combinations(mesh.triangles, 2):
        ax.set_xlim(rng), ax.set_ylim(rng), ax.set_zlim(rng)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        ax.scatter(tri1[:,0], tri1[:,1], tri1[:,2], color='r')
        ax.scatter(tri2[:,0], tri2[:,1], tri2[:,2], color='b')
        is_coplanar = trimesh.triangles.all_coplanar([tri1, tri2])
        normals = trimesh.triangles.normals([tri1, tri2])[0]
        normal1, normal2 = normals[0], normals[1]
        angle = np.arccos(np.dot(normal1, normal2))
        eps = 0.01
        is_parallel = (angle < eps) | (np.pi -eps < angle)
        print(f"{~is_coplanar & is_parallel}")
        print(f"norm1:{normal1}, norm2:{normal2}, angle:{angle}")
        ax.clear()
    print()
    mesh.show()
    input()

if __name__ == "__main__":
    main()