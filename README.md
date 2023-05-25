# Install

```bash
mamba install compilers make cmake qhull pybind11
python setup.py install
```

# Usage

```python
import trimesh
from trimesh import creation
import pyvista as pv
# create test points
points0 = creation.box().sample(10000)
points1 = creation.uv_sphere(2).sample(10000)

# high level interface
from pyibs import IBS
ibs = IBS(points0,points1)
ibs_mesh = ibs.mesh
ibs_points = ibs.points
pl = pv.Plotter()
pl.add_mesh(points0,'r')
pl.add_mesh(points1,'b')
pl.add_mesh(ibs_mesh,'y',opacity=0.5)
pl.show()

# low level interface
import libibs
import numpy as np
n0,n1 = len(points0),len(points1)
points = np.concatenate([points0,points1]).astype('float32')
ids = np.zeros(n0+n1).astype('int32')
ids = np.zeros(n0+n1).astype('int32')
ids[n0:] = 1
ibs_v,ibs_f,ibs_irs = libibs.create_ibs(points,ids)
ibs_mesh = trimesh.Trimesh(ibs_v,ibs_f)
pl = pv.Plotter()
pl.add_mesh(points0,'r')
pl.add_mesh(points1,'b')
pl.add_mesh(ibs_mesh,'y',opacity=0.5)
pl.show()