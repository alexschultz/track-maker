from collada import *
import os

mesh = Collada(os.path.join('input-mesh-files', 'Canad_track_road.dae'))

geo = mesh.geometries[0]
prims = geo.primitives
print('thing')
