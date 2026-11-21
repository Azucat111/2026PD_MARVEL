import numpy as np,pytest
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery

@pytest.mark.parametrize('frame',[GeometryFrame.extended(width_cells=80,height_cells=80),GeometryFrame.marvel_native(width_cells=50,height_cells=60,origin_x=-3.2,origin_y=-7.6)])
def test_vector_containing_cells_match_scalar_at_lattice_boundaries(frame):
 converter=getattr(frame,'world_to_cells_floor',None)
 assert callable(converter), 'Batch containing-cell conversion missing'
 rng=np.random.default_rng(4360);points=rng.uniform(-20,90,size=(1000,2))
 boundary=np.array([frame.cell_to_world([i,i]) for i in range(10)])
 points=np.vstack([points,boundary,np.nextafter(boundary,np.inf),np.nextafter(boundary,-np.inf)])
 assert np.array_equal(converter(points),[frame.world_to_cell_floor(p) for p in points])
 assert np.array_equal(converter(points,clip=True),[frame.clip_cell(*frame.world_to_cell_floor(p)) for p in points])

def test_ray_uses_batch_transform_without_scalar_lookup_per_sample():
 class CountingFrame(GeometryFrame):
  scalar_calls=0
  def world_to_cell_floor(self,p):
   type(self).scalar_calls+=1
   return super().world_to_cell_floor(p)
 frame=CountingFrame(0.,0.,1.,80,80);rec=ObservedSearchRecovery(frame)
 cells=list(rec._ray_cells(np.array([3.2,4.1]),np.array([35.2,64.1])))
 assert len(cells)>100
 assert CountingFrame.scalar_calls==0, 'Ray still transforms each sample separately'


def test_radius_point_two_probe_still_rejects_adjacent_obstacle():
 frame=GeometryFrame.extended(width_cells=8,height_cells=8);rec=ObservedSearchRecovery(frame)
 rec.cells={(c,r):0 for c in range(8) for r in range(8)};rec.cells[4,3]=1
 assert not rec.segment_safe([3.81,3.5],[3.9,3.5])
