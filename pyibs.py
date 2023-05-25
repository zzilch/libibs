import libibs
import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi,Delaunay
from scipy.special import softmax

class IBS():
    def __init__(self,points0,points1,n=500,use_angle=True,use_dist=True,dist_power=2,clip='sphere',bounded=True) -> None:
        # input
        self.points0 = points0
        self.points1 = points1
        
        # output mesh
        self.n = n
        self.sphere = None
        self.mesh = None
        self.ir0 = None
        self.ir1 = None
        self.clip = clip
        self.bounded = bounded
        
        # output points
        self.use_angle = use_angle
        self.use_dist = use_dist
        self.dist_power = dist_power
        self.w_area = None
        self.w_angle = None
        self.w_dist = None
        self.weights = None
        self.points = None
        self.points_tri = None
        
        if self.bounded:
            self.create_mesh_bounded()
        else:
            self.create_mesh()
        self.sample_points()
    
    def create_mesh(self):
        n0 = len(self.points0)
        n1 = len(self.points1)

        points = np.concatenate([
            self.points0,
            self.points1]).astype('float32')
        ids = np.zeros(n0+n1).astype('int32')
        ids[n0:] = 1

        self.bounds = np.stack([
            points.min(0),
            points.max(0)],axis=-1).reshape(-1)

        self.extent = self.bounds[1::2]-self.bounds[::2]
        self.center = (self.bounds[1::2]+self.bounds[::2])/2
        self.length = np.linalg.norm(self.extent)/2

        clip_bounds = np.array([
            self.center[0],
            self.center[1],
            self.center[2],
            self.length])

        v,f,_ = libibs.create_ibs(np.concatenate([points]),ids)

        ibs = pv.make_tri_mesh(v,f)

        if self.clip=='sphere':
            bound_mesh = pv.Sphere(self.length,self.center)
            ibs = ibs.clip_surface(bound_mesh,invert=True)
        elif self.clip=='delaunay':
            bound_mesh = pv.wrap(points).delaunay_3d()
            ibs = ibs.clip_surface(bound_mesh,invert=True)
        elif self.clip=='box':
            ibs = ibs.clip_box(clip_bounds,invert=True)
        # ibs = ibs.triangulate()

        self.mesh = trimesh.Trimesh(ibs.points,ibs.faces.reshape(-1,4)[:,1:],process=False)
        centers = self.mesh.triangles_center

        kdtree0 = KDTree(self.points0)
        kdtree1 = KDTree(self.points1)

        d0,ir0 = kdtree0.query(centers,workers=-1)
        d1,ir1 = kdtree1.query(centers,workers=-1)

        self.d = (d0+d1)/2
        self.ir0 = ir0
        self.ir1 = ir1

    def create_mesh_bounded(self):
        n0 = len(self.points0)
        n1 = len(self.points1)

        points = np.concatenate([
            self.points0,
            self.points1]).astype('float32')
        
        self.bounds = np.stack([
            points.min(0),
            points.max(0)],axis=-1).reshape(-1)

        self.extent = self.bounds[1::2]-self.bounds[::2]
        self.center = (self.bounds[1::2]+self.bounds[::2])/2
        self.length = np.linalg.norm(self.extent)/2

        n2 = (n0+n1)//10
        shell = fibonacci_sphere(n2)
        # shell = shell*self.length*1.5+self.center
        shell = shell*self.length+self.center
        self.shell = shell
        
        points = np.concatenate([points,shell])
        ids = np.zeros(n0+n1+n2).astype('int32')
        ids[n0:] = 1
        ids[n0+n1:] = 2

        v,f,p = libibs.create_ibs(np.concatenate([points]),ids)
        f = f[~(p>=n0+n1).any(axis=-1)]

        ibs = pv.make_tri_mesh(v,f)

        self.mesh = trimesh.Trimesh(ibs.points,ibs.faces.reshape(-1,4)[:,1:],process=False)
        self.mesh.remove_unreferenced_vertices()
        centers = self.mesh.triangles_center

        kdtree0 = KDTree(self.points0)
        kdtree1 = KDTree(self.points1)

        d0,ir0 = kdtree0.query(centers,workers=-1)
        d1,ir1 = kdtree1.query(centers,workers=-1)

        self.d = (d0+d1)/2
        self.ir0 = ir0
        self.ir1 = ir1

    def sample_points(self):
        centers = self.mesh.triangles_center
        vec0 = self.points0[self.ir0]-centers
        norms = self.mesh.face_normals

        cos = np.abs(np.sum((vec0*norms)/((np.linalg.norm(vec0,axis=-1)*np.linalg.norm(norms,axis=-1))[:,None]+np.finfo(float).eps),axis=-1))
        alpha = np.arccos(cos)
        self.alpha = alpha

        w_area = self.mesh.area_faces
        w_angle = np.clip(1-alpha/(np.pi/2),0,1)
        dmax = self.d.max()
        dmin = self.d.min()
        w_dist = (dmax-self.d)/(dmax-dmin)
        if self.dist_power == 'softmax':
            w_dist = softmax(w_dist)
        else:
            w_dist = np.power(w_dist,self.dist_power)
        # w_dist = 1-np.power(self.d/self.length,self.dist_power)
        # w_dist = np.exp(-(self.d/self.length)**2/(2*self.dist_power**2)) 

        weights = w_area 
        if self.use_angle: weights = weights * w_angle
        if self.use_dist: weights = weights * w_dist  

        self.weights = weights
        self.w_area = w_area
        self.w_angle = w_angle
        self.w_dist = w_dist

        self.points, self.points_tri = self.mesh.sample(self.n,face_weight=self.weights,return_index=True)
        return self.points

    def get_ir_points(self,idx=0,type='mesh'):
        if idx==0:
            points = self.points0
            ir = self.ir0
        else:
            points = self.points1
            ir = self.ir1
        
        if type=='points':
            ir = ir[self.points_tri]
        return points[ir]

    def sample_line(self,count=500,type='mesh'):
        ir0 = self.get_ir_points(0,type=type)
        ir1 = self.get_ir_points(1,type=type)
        l = np.linalg.norm(ir1-ir0,axis=-1)
        if self.dist_power == 'softmax':
            w_l = softmax(1-l/self.length)
        else:
            w_l = np.power(1-l/self.length,self.dist_power)
        return sample_line(
            ir0,
            ir1,
            count,
            w_l
        )
    
    def sample_delaunay(self,count=500,type='mesh'):
        return sample_delaunay(
            np.concatenate([
                self.get_ir_points(0,type=type),
                self.get_ir_points(1,type=type),
            ]),
            count
        )
        
def fibonacci_sphere(n=48,offset=False):
    """Sample points on sphere using fibonacci spiral.

    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    :param int n: number of sample points, defaults to 48
    :param bool offset: set True to get more uniform samplings when n is large , defaults to False
    :return array: points samples
    """

    golden_ratio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / golden_ratio

    if offset:
        if n >= 600000:
            epsilon = 214
        elif n>= 400000:
            epsilon = 75
        elif n>= 11000:
            epsilon = 27
        elif n>= 890:
            epsilon = 10
        elif n>= 177:
            epsilon = 3.33
        elif n>= 24:
            epsilon = 1.33
        else:
            epsilon = 0.33
        phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    else:
        phi = np.arccos(1 - 2*(i+0.5)/n)

    x = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],axis=-1)
    return x

def create_ibs_python(x0,x1,bounds=None):
    """create ibs from two point cloud

    Args:
        x0 ([type]): point cloud 0
        x1 ([type]): point cloud 2
        bounds ([type], optional): (xmin,xmax,ymin,ymax,zmin,zmax) Defaults to None.

    Returns:
        Trimesh: ibs surface
    """
    n0,n1 = len(x0),len(x1)
    x01 = np.concatenate([x0,x1])
    
    vor =  Voronoi(x01)
    # find ridges
    ridge_idx = np.where((vor.ridge_points<n0).sum(-1)==1)[0]
    # remove points at infinity
    ridge_idx = [i for i in ridge_idx if -1 not in vor.ridge_vertices[i]]
    # create ridge polygons
    polys = np.asarray(vor.ridge_vertices,dtype=object)[ridge_idx]
    polys = np.concatenate(list(map(lambda x:[len(x),]+x,polys)))

    if bounds is None: 
        bounds = np.stack([x01.min(0),x01.max(0)],axis=-1).reshape(-1)
    ibs = pv.PolyData(vor.vertices,polys)
    ibs = ibs.clip_box(bounds.tolist(),invert=False)
    ibs = ibs.triangulate()
    ibs = trimesh.Trimesh(ibs.points,ibs.cells.reshape(-1,4)[:,1:4])
    return ibs

def sample_line(points_from,points_to,count=500,line_weight=None):
    if line_weight is None:
        line_weight = np.linalg.norm(points_to-points_from,axis=-1)

    weight_cum = np.cumsum(line_weight)
    line_pick = np.random.random(count) * weight_cum[-1]
    line_index = np.searchsorted(weight_cum, line_pick)

    line_origins = points_from
    line_vectors = points_to-points_from

    line_origins = line_origins[line_index]
    line_vectors = line_vectors[line_index]

    random_lengths = np.random.random(len(line_vectors))
    sample_vector = line_vectors * random_lengths[:,None]

    samples = sample_vector + line_origins
    return samples

def sample_delaunay(points,count=500):
    dln = Delaunay(points)
    hull = trimesh.Trimesh(dln.points,dln.convex_hull)
    samples = trimesh.sample.volume_mesh(hull,count)
    return samples