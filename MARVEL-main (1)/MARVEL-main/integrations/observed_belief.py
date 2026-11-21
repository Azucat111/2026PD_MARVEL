"""Conservative sensor-only conversion into MARVEL's belief lattice.
Occupied physical cells dominate; FREE requires every positively overlapping
physical cell observed FREE. No truth-map or obstacle-manager input.
"""
import numpy as np
from parameter import FREE,OCCUPIED,UNKNOWN
from integrations.gppo.search_recovery import ObservedSearchRecovery

class ObservedBeliefMap:
    def __init__(self,frame,origin,resolution,shape):
        self.frame=frame
        self.evidence=ObservedSearchRecovery(frame)
        self.shape=tuple(shape)
        self.ranges=[]
        for axis,size in enumerate((shape[1],shape[0])):
            lo=(float(origin[axis])+np.arange(size)*resolution-frame.origin[axis])/frame.cell_size
            hi=lo+resolution/frame.cell_size
            # Half-open cells: an exact boundary has no positive overlap.
            first=np.floor(lo).astype(int)
            last=np.floor(np.nextafter(hi,-np.inf)).astype(int)
            self.ranges.append((first,last))

    def observe(self,observations):
        self.evidence.observe(observations)

    def point_clear(self,point):
        return self.evidence.segment_safe(point,point)

    def rasterize(self):
        physical=np.full((self.frame.height_cells,self.frame.width_cells),-1,dtype=np.int8)
        if self.evidence.cells:
            keys=np.asarray(list(self.evidence.cells),dtype=int)
            values=np.fromiter(self.evidence.cells.values(),dtype=np.int8)
            physical[keys[:,1],keys[:,0]]=values
        x0,x1=self.ranges[0]; y0,y1=self.ranges[1]
        occupied=np.zeros(self.shape,dtype=bool)
        free=np.ones(self.shape,dtype=bool)
        for xs in (x0,x1):
            for ys in (y0,y1):
                valid=(ys[:,None]>=0)&(ys[:,None]<self.frame.height_cells)&(xs[None,:]>=0)&(xs[None,:]<self.frame.width_cells)
                values=physical[np.clip(ys,0,self.frame.height_cells-1)[:,None],np.clip(xs,0,self.frame.width_cells-1)[None,:]]
                occupied |= valid&(values==1)
                free &= valid&(values==0)
        result=np.full(self.shape,UNKNOWN,dtype=np.int32)
        result[free]=FREE
        result[occupied]=OCCUPIED
        return result
