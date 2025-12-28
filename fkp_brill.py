import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from itertools import combinations


#Gitter
a = 1
a_1 =  np.array([ 1,  0, 0])
a_2 =  np.array([ -0.5,  np.sqrt(3)/2, 0])
a_3 =  np.array([ 0,  0, 4])


#reziprokes Gitter 
nenner = (np.dot(a_1, np.cross(a_2, a_3))) / (2 * np.pi)
b_1 = np.cross(a_2, a_3) / nenner
b_2 = np.cross(a_3, a_1) / nenner
b_3 = np.cross(a_1, a_2) / nenner


def plane(P, Q):
    n = Q - P        
    d = np.dot(n, P)         
    return n, d

def side_of_plane(X, n, d, tol=1e-9):
    # > 0 : seite der Normalen
    # 0 : auf Ebene
    # < 0 : andere Seite
    X = np.asarray(X, dtype=float)
    value = np.dot(n, X) - d
    if abs(value) < tol:
        return 0
    return value


points = []

#Anzahl an Iterationen
N = 1

for i in range(-N, N+1):
    for j in range(-N, N+1):
        for k in range(-N, N+1):
            points.append(i*b_1 + j*b_2 + k*b_3)
            
points = np.array(points)



#Mittelpunkte zwischen dem Ursprung und entsprechendem Punkt finden
mid_points = []
center = np.array([0,0,0])


for point in points:
    if not np.allclose(point, center):
        mid_points.append(point/2)
mid_points = np.array(mid_points)

# Ebene konstruieren und falls Punkt auf der anderen Seite liegt, muss dieser entfernt werden.
# tol muss verwendet werden, weil floats ... 

tol = 1e-6

def remove_points(P, liste_points):
    n, d = plane(P, center)
    side_test = side_of_plane(center, n, d, tol)
    
    temp = []
    
    for element in liste_points:
        side_other = side_of_plane(element, n, d, tol)

        if side_other * side_test >= -tol:
            temp.append(element)

    
    return np.array(temp)

# In der Liste ist der Punkt selbst enthalten, was aber kein Problem ist, solange es keinen Rundungsfehler gibt. 

temp_points = mid_points.copy()

for point in mid_points:
    temp_points = remove_points(point, temp_points)

temp_points = np.unique(np.round(temp_points, decimals=8), axis=0)
cleaned_mid_points = temp_points


mid_points = np.array(cleaned_mid_points)



#Ebene bauen.

ebenen = []

for mid in mid_points:
    G = 2 * mid
    d = np.dot(G, mid) 
    ebenen.append((G, d))
    


vertices = []
tol2 = 1e-6

for(n1, d1), (n2, d2), (n3, d3) in combinations(ebenen, 3):

    A = np.vstack([n1, n2, n3])
    b = np.array([d1, d2, d3])

    if abs(np.linalg.det(A)) < tol:
        continue

    k = np.linalg.solve(A, b)

    inside = True
    for n, d in ebenen:
        if np.dot(n, k) > d + tol:
            inside = False
            break
    if inside:
        vertices.append(k)



vertices = np.unique(np.round(vertices, 8), axis=0)


hull = ConvexHull(vertices)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


faces = [vertices[simplex] for simplex in hull.simplices]

poly = Poly3DCollection(faces, facecolor='lightblue', edgecolor='k', alpha=0.7)

ax.add_collection3d(poly)

ax.scatter(points[:,0], points[:,1], points[:,2], s=60, color='blue')
ax.scatter(mid_points[:,0], mid_points[:,1], mid_points[:,2], s=60, color='orange')

ax.scatter(0, 0, 0, color='red', s=80)  # Ursprung

# gleiche Achsenskalierung
max_range = np.max(np.ptp(points, axis=0))
mid = points.mean(axis=0)

ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('kz')

plt.tight_layout()
plt.show()
