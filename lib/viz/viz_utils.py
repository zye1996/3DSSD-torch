import os

import numpy as np
import scipy
from scipy.spatial import Delaunay

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)



def point_viz(pts, name='demo', dump_dir='./data_viz_dump'):
    # if os.path.exists(dump_dir):
    #     os.system('rm -r %s' % dump_dir)
    # os.mkdir(dump_dir)

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    pts = pts[:, 0:3]

    # Dump OBJ files for the colored point cloud
    # pc = pc_util.flip_axis_to_viz(pts)
    write_ply_pc(pts, os.path.join(dump_dir, '%s.obj' % name))


def write_ply_pc(points, out_filename, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as OBJ file """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        fout.write('v %f %f %f\n' % (points[i,0],points[i,1],points[i,2]))
    fout.close()