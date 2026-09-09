"""Accelerated multi-epoch capture, evaluation, logging, and visualization."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
warnings.filterwarnings("ignore",message=r"Unable to import Axes3D\..*",category=UserWarning,module=r"matplotlib\.projections")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from simlab.utils.paths import PROJECT_ROOT
from .clock import EnvironmentTimeManager
from .config import TemporalConfig
from .evaluation import TemporalImageMatcher
from .renderer import TemporalSceneRenderer
from .state import EnvironmentStateFactory, TemporalEventScheduler
from .trajectory import FixedWingTrajectoryManager


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _feature_metrics(image: np.ndarray) -> dict[str,float|int]:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); hist=cv2.calcHist([gray],[0],None,[256],[0,256]).ravel(); p=hist[hist>0]/gray.size
    corners=cv2.goodFeaturesToTrack(gray,1000,.01,5); edges=cv2.Canny(gray,60,150)
    return {"texture_entropy":float(-(p*np.log2(p)).sum()),"edge_density":float(np.mean(edges>0)),"corner_density":0. if corners is None else float(len(corners)/gray.size),"brightness":float(gray.mean()),"blur_score":float(cv2.Laplacian(gray,cv2.CV_64F).var())}


def _write_video(path:Path,frames:list[np.ndarray],fps:float)->None:
    if not frames:return
    height,width=frames[0].shape[:2];writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened():raise RuntimeError(f"cannot create video {path}")
    for frame in frames:writer.write(frame)
    writer.release()


class TemporalBatchRunner:
    def __init__(self,cfg:TemporalConfig):
        self.cfg=cfg; cv2.setRNGSeed(cfg.random_seed); cv2.setNumThreads(1)
        self.clock=EnvironmentTimeManager(cfg.time.days_per_real_second,cfg.time.maximum_year)
        self.scheduler=TemporalEventScheduler(cfg.events,cfg.random_seed)
        self.states=EnvironmentStateFactory(cfg,self.scheduler); self.renderer=TemporalSceneRenderer(cfg); self.matcher=TemporalImageMatcher()

    def _directory(self)->Path:
        root=Path(self.cfg.output_directory); root=root if root.is_absolute() else PROJECT_ROOT/root
        out=root/datetime.now().strftime("run_%Y%m%d_%H%M%S"); i=1
        while out.exists(): out=root/f"{out.name}_{i:02d}"; i+=1
        for name in ("reference_images","query_images","figures","epoch_usd_layers"): (out/name).mkdir(parents=True,exist_ok=True)
        return out

    @staticmethod
    def _write_usda(path:Path,state:dict[str,Any])->None:
        fields="\n".join(f'        custom string temporal:{k} = "{v}"' for k,v in state.items() if isinstance(v,(str,int,float)))
        path.write_text(f'''#usda 1.0\n(\n    defaultPrim = "World"\n)\ndef Xform "World" {{\n    def Xform "Environment" {{\n{fields}\n    }}\n}}\n''',encoding="utf-8")

    def run(self)->Path:
        out=self._directory(); timeline=[]; truth_rows=[]; pose_rows=[]; matching=[]; reference_images:list[np.ndarray]=[];poses=FixedWingTrajectoryManager(self.cfg.camera).poses()
        reference_features:list[dict[str,float|int]]=[]
        for epoch_index,epoch in enumerate(self.cfg.epochs):
            time=self.clock.jump_to_epoch(epoch.year); state=self.states.create(epoch); state_row=asdict(state)
            timeline.append({"virtual_time":time.virtual_day,"virtual_year":time.virtual_year,"season":state.season,"weather":state.weather,"global_change_score":state.environment_change_score,"vegetation_change_score":state.vegetation_change_score,"building_change_score":state.building_change_score,"road_appearance_change_score":state.road_appearance_change_score,"road_geometry_change_score":state.road_geometry_change_score,"season_change_score":state.season_change_score,"weather_change_score":state.weather_change_score,"illumination_change_score":state.illumination_change_score,"structural_change_score":state.structural_change_score,"appearance_change_score":state.appearance_change_score})
            truth_rows.append({**state_row,"applied_event_ids":";".join(state.applied_event_ids)})
            world=self.renderer.render_world(state); views=self.renderer.capture(world,poses)
            if self.cfg.capture.save_epoch_usd_layers:self._write_usda(out/"epoch_usd_layers"/f"{epoch.name}.usda",state_row)
            for pose_index,(view,pose) in enumerate(zip(views,self.renderer.camera_poses())):
                pose_rows.append({"epoch":epoch.name,**pose})
                filename=f"pose_{pose_index:03d}.png"
                if epoch_index==0:
                    reference_images.append(view.copy()); reference_features.append(_feature_metrics(view))
                    if self.cfg.capture.save_images:cv2.imwrite(str(out/"reference_images"/filename),view)
                else:
                    epoch_dir=out/"query_images"/epoch.name; epoch_dir.mkdir(exist_ok=True)
                    if self.cfg.capture.save_images:cv2.imwrite(str(epoch_dir/filename),view)
                    selected,metrics=self.matcher.retrieve(view,reference_images); current_features=_feature_metrics(view)
                    row={"epoch":epoch.name,"virtual_year":epoch.year,"trajectory_index":pose_index,"selected_reference_index":selected,"is_correct_place_match":selected==pose_index,"false_place_match":selected>=0 and selected!=pose_index,**metrics,**current_features}
                    row.update({"environment_change_score":state.environment_change_score,"structural_change_score":state.structural_change_score,"appearance_change_score":state.appearance_change_score,"vegetation_change_score":state.vegetation_change_score,"building_change_score":state.building_change_score,"road_appearance_change_score":state.road_appearance_change_score,"road_geometry_change_score":state.road_geometry_change_score,"season_change_score":state.season_change_score,"weather_change_score":state.weather_change_score})
                    row.update({f"delta_{k}":float(current_features[k])-float(reference_features[pose_index][k]) for k in current_features})
                    matching.append(row)
            if self.cfg.capture.save_images:
                video_path=(out/"reference_images"/"spiral_flight.mp4") if epoch_index==0 else (out/"query_images"/epoch.name/"spiral_flight.mp4")
                _write_video(video_path,views,self.cfg.camera.capture_fps)
        _write_csv(out/"environment_timeline.csv",timeline); _write_csv(out/"environment_change_ground_truth.csv",truth_rows); _write_csv(out/"camera_poses.csv",pose_rows); _write_csv(out/"matching_results.csv",matching)
        with (out/"environment_events.jsonl").open("w",encoding="utf-8") as stream:
            for event in self.scheduler.replay_record():stream.write(json.dumps(event)+"\n")
        scores=np.array([r["environment_change_score"] for r in matching],float); inliers=np.array([r["inlier_ratio"] for r in matching],float); correct=np.array([r["is_correct_place_match"] for r in matching],float)
        correlation=float(np.corrcoef(scores,inliers)[0,1]) if len(set(scores))>1 and len(set(inliers))>1 else float("nan")
        summary=[{"metric":"query_image_count","value":len(matching)},{"metric":"correct_place_match_rate","value":float(correct.mean())},{"metric":"false_match_rate","value":float(np.mean([r["false_place_match"] for r in matching]))},{"metric":"mean_inlier_ratio","value":float(inliers.mean())},{"metric":"environment_change_vs_inlier_pearson","value":correlation}]
        _write_csv(out/"summary_metrics.csv",summary)
        fig,ax=plt.subplots(figsize=(6,4));ax.scatter(scores,inliers,c=[r["structural_change_score"] for r in matching],cmap="viridis");ax.set(xlabel="Environment change score",ylabel="RANSAC inlier ratio",title="Temporal change vs visual matching");fig.colorbar(ax.collections[0],ax=ax,label="Structural change");fig.tight_layout();fig.savefig(out/"figures"/"environment_change_vs_inlier_ratio.png",dpi=180);plt.close(fig)
        by_epoch={e.name:[r for r in matching if r["epoch"]==e.name] for e in self.cfg.epochs[1:]}; fig,ax=plt.subplots(figsize=(8,4));ax.bar(list(by_epoch),[np.mean([r["is_correct_place_match"] for r in rows]) for rows in by_epoch.values()]);ax.tick_params(axis="x",rotation=35);ax.set(ylabel="Correct place match rate",ylim=(0,1.05));fig.tight_layout();fig.savefig(out/"figures"/"matching_accuracy_by_epoch.png",dpi=180);plt.close(fig)
        with (out/"resolved_config.yaml").open("w",encoding="utf-8") as stream:yaml.safe_dump(asdict(self.cfg),stream,sort_keys=False)
        (out/"run_log.txt").write_text(f"seed={self.cfg.random_seed}\nepochs={len(self.cfg.epochs)}\nquery_images={len(matching)}\n",encoding="utf-8")
        return out
