"""Repeatable fixed-wing patrol trajectories in metric map coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FlightPose:
    index:int
    x_m:float
    y_m:float
    z_m:float
    yaw_deg:float
    progress:float


class FixedWingTrajectoryManager:
    def __init__(self,camera_cfg):self.cfg=camera_cfg
    def poses(self)->tuple[FlightPose,...]:
        c=self.cfg
        if c.trajectory_mode=="fixed":
            points=[(float(x),float(y)) for x,y in c.trajectory_xy]
        else:
            points=[];direction=-1. if c.clockwise else 1.
            for i in range(c.spiral_samples):
                progress=i/(c.spiral_samples-1);radius=c.spiral_start_radius_m+progress*(c.spiral_end_radius_m-c.spiral_start_radius_m);angle=direction*2*math.pi*c.spiral_turns*progress;x=c.spiral_center_xy_m[0]+radius*math.cos(angle);y=c.spiral_center_xy_m[1]+radius*math.sin(angle);points.append((x,y))
        poses=[]
        for i,(x,y) in enumerate(points):
            if i+1<len(points):nx,ny=points[i+1]
            else:nx,ny=x+(x-points[i-1][0]),y+(y-points[i-1][1])
            yaw=math.degrees(math.atan2(ny-y,nx-x));poses.append(FlightPose(i,x,y,c.altitude_m,yaw,i/max(1,len(points)-1)))
        return tuple(poses)
