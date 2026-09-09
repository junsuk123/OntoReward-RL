"""Fast deterministic image backend mirroring the mutable USD scene semantics."""

from __future__ import annotations

import cv2
import numpy as np

from .config import TemporalConfig
from .state import EnvironmentState
from .trajectory import FixedWingTrajectoryManager


class TemporalSceneRenderer:
    BASE_SIZE = 960
    SEASON_GROUND = {"spring": (72, 125, 76), "summer": (55, 112, 57), "autumn": (60, 105, 128), "winter": (192, 192, 184)}
    SEASON_LEAF = {"spring": (62, 150, 70), "summer": (38, 118, 45), "autumn": (35, 105, 190), "winter": (115, 125, 120)}
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg; rng = np.random.default_rng(cfg.random_seed)
        # Stable valid vegetation zones, never inside the central road/buildings.
        zones = [(70,330,80,360), (630,890,70,360), (70,330,600,880), (630,890,600,880)]
        positions=[]
        for i in range(cfg.changes.tree_initial_count+cfg.changes.tree_new_count):
            x0,x1,y0,y1=zones[i%len(zones)]; positions.append((int(rng.integers(x0,x1)),int(rng.integers(y0,y1))))
        self.tree_positions=tuple(positions)

    def render_world(self, state: EnvironmentState) -> np.ndarray:
        n=self.BASE_SIZE; image=np.full((n,n,3), self.SEASON_GROUND[state.season], np.uint8)
        rng=np.random.default_rng(self.cfg.random_seed)
        noise=rng.normal(0,7,(n,n,1)); image=np.clip(image.astype(float)+noise,0,255).astype(np.uint8)
        # Large-map parcels and secondary roads provide persistent landmarks.
        parcels=((18,18,300,260,(68,122,72)),(640,22,932,280,(82,132,78)),(25,665,300,932,(58,112,92)),(650,650,935,935,(90,120,68)))
        for x0,y0,x1,y1,color in parcels:cv2.rectangle(image,(x0,y0),(x1,y1),color,-1);cv2.rectangle(image,(x0,y0),(x1,y1),(38,68,42),3)
        for x in (285,675):cv2.rectangle(image,(x-24,0),(x+24,n),(88,88,88),-1);cv2.line(image,(x,0),(x,n),(210,210,210),3)
        road_half=round(105*state.road_width_scale); road_color=int(72+58*state.road_appearance_change_score)
        cv2.rectangle(image,(0,n//2-road_half),(n,n//2+road_half),(road_color,road_color,road_color),-1)
        line=int(245*state.road_marking_visibility)
        for x in range(-40,n,90): cv2.rectangle(image,(x,n//2-4),(x+45,n//2+4),(line,line,line),-1)
        # One persistent building and one temporal construction site.
        cv2.rectangle(image,(350,100),(565,320),(105,100,95),-1); cv2.rectangle(image,(370,120),(545,300),(150,145,140),-1)
        for x in range(385,540,38):
            for y in range(135,295,38): cv2.rectangle(image,(x,y),(x+17,y+17),(40,55,65),-1)
        if state.building_variant != "EmptyLot":
            cv2.rectangle(image,(365,650),(610,880),(80,80,84),3)
            if state.building_variant == "Construction":
                for x in range(380,610,38): cv2.line(image,(x,655),(x,875),(125,125,130),4)
                for y in range(670,880,42): cv2.line(image,(370,y),(605,y),(125,125,130),4)
            else:
                color=(122,112,105) if state.building_variant=="Completed" else (155,105,80)
                cv2.rectangle(image,(375,660),(600,870),color,-1)
                for x in range(395,590,43):
                    for y in range(680,860,43): cv2.circle(image,(x,y),8,(35,45,58),-1)
        for i,(x,y,sx,sy) in enumerate(((105,105,52,38),(205,185,44,62),(755,120,58,45),(850,220,42,70),(110,755,68,42),(220,855,48,52),(725,770,50,65),(850,850,72,44),(445,80,55,42),(520,880,62,38),(80,445,42,58),(870,520,55,48))):
            shade=95+(i%4)*18;cv2.rectangle(image,(x-sx//2,y-sy//2),(x+sx//2,y+sy//2),(shade,shade-5,shade-10),-1);cv2.rectangle(image,(x-sx//2,y-sy//2),(x+sx//2,y+sy//2),(45,45,45),2)
        leaf=self.SEASON_LEAF[state.season]; radius=max(4,round(10*state.tree_scale))
        for index,(x,y) in enumerate(self.tree_positions[:state.visible_tree_count]):
            cv2.circle(image,(x,y),max(3,radius//3),(55,60,65),-1); cv2.circle(image,(x,y),radius,leaf,-1); cv2.circle(image,(x-radius//3,y-radius//3),max(2,radius//3),(leaf[0]+10,min(255,leaf[1]+25),min(255,leaf[2]+10)),-1)
        image=cv2.convertScaleAbs(image,alpha=state.lighting_scale,beta=0)
        if state.weather=="fog": image=cv2.addWeighted(image,.52,np.full_like(image,205),.48,0)
        elif state.weather=="rain":
            for _ in range(900):
                x,y=rng.integers(0,n,2); cv2.line(image,(x,y),(x+3,y+13),(185,185,175),1)
        elif state.weather=="snow":
            for _ in range(1200):
                x,y=rng.integers(0,n,2); cv2.circle(image,(x,y),int(rng.integers(1,4)),(245,245,245),-1)
        elif state.weather=="cloudy": image=cv2.addWeighted(image,.75,np.full_like(image,135),.25,0)
        return cv2.resize(image,(self.cfg.map.canvas_px,self.cfg.map.canvas_px),interpolation=cv2.INTER_LINEAR)

    def capture(self, world: np.ndarray, poses=None) -> list[np.ndarray]:
        width,height=self.cfg.camera.resolution; views=[]; n=self.cfg.map.canvas_px;poses=poses or FixedWingTrajectoryManager(self.cfg.camera).poses()
        for pose in poses:
            cx=int(n/2+pose.x_m/self.cfg.map.extent_m*n);cy=int(n/2-pose.y_m/self.cfg.map.extent_m*n)
            view=world[cy-height//2:cy+height//2,cx-width//2:cx+width//2]
            if view.shape[:2] != (height,width): view=cv2.resize(view,(width,height))
            views.append(view.copy())
        return views

    def camera_poses(self) -> list[dict[str,float|str]]:
        return [{"trajectory_index":p.index,"x_m":p.x_m,"y_m":p.y_m,"z_m":p.z_m,"roll_deg":0.,"pitch_deg":0.,"yaw_deg":p.yaw_deg,"trajectory_progress":p.progress} for p in FixedWingTrajectoryManager(self.cfg.camera).poses()]
