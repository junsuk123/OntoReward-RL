"""Isaac Sim 5.1 adapter for temporal USD mutation and nadir RGB capture."""

from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
import time

import cv2
import numpy as np

from .config import DEFAULT_TEMPORAL_CONFIG,load_temporal_config
from .state import EnvironmentStateFactory,TemporalEventScheduler
from .evaluation import TemporalImageMatcher
from .trajectory import FixedWingTrajectoryManager


def _match_visualization(reference,query,title:str,limit:int=80):
    """Reference/query correspondence view with green inliers and red outliers."""
    orb=cv2.ORB_create(nfeatures=1200);rkp,rd=orb.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),None);qkp,qd=orb.detectAndCompute(cv2.cvtColor(query,cv2.COLOR_BGR2GRAY),None)
    h=max(reference.shape[0],query.shape[0]);w=reference.shape[1]+query.shape[1];canvas=np.zeros((h+42,w,3),np.uint8);canvas[42:42+reference.shape[0],:reference.shape[1]]=reference;canvas[42:42+query.shape[0],reference.shape[1]:]=query;cv2.putText(canvas,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2,cv2.LINE_AA)
    if rd is None or qd is None:return canvas
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(qd,rd,k=2);good=[a for a,b in pairs if a.distance<.78*b.distance];mask=np.zeros(len(good),bool)
    if len(good)>=4:
        q=np.float32([qkp[m.queryIdx].pt for m in good]);r=np.float32([rkp[m.trainIdx].pt for m in good]);_,found=cv2.findHomography(q,r,cv2.RANSAC,4.);mask=found.ravel().astype(bool) if found is not None else mask
    for i,m in enumerate(good[:limit]):
        q=tuple(round(v) for v in qkp[m.queryIdx].pt);r=tuple(round(v) for v in rkp[m.trainIdx].pt);color=(50,220,70) if mask[i] else (40,55,235);q=(q[0]+reference.shape[1],q[1]+42);r=(r[0],r[1]+42);cv2.line(canvas,r,q,color,1,cv2.LINE_AA);cv2.circle(canvas,r,2,color,-1);cv2.circle(canvas,q,2,color,-1)
    return canvas


class IsaacTemporalScene:
    """Stable USD hierarchy whose mutable attributes are updated between epochs."""
    def __init__(self,cfg):
        import omni.usd
        from pxr import Gf,Sdf,UsdGeom,UsdLux
        self.cfg=cfg;self.stage=omni.usd.get_context().get_stage();self.Gf=Gf;self.UsdGeom=UsdGeom
        for path in ("/World","/World/Environment","/World/Environment/Terrain","/World/Environment/Roads","/World/Environment/Buildings","/World/Environment/Vegetation","/World/Environment/Weather","/World/Lighting","/World/UAV","/World/GroundTruth"):UsdGeom.Xform.Define(self.stage,path)
        extent=cfg.map.extent_m
        self.ground=self._cube("/World/Environment/Terrain/Ground",(0,0,-.15),(extent,extent,.3),(0.25,.45,.25),"terrain_ground")
        self.road=self._cube("/World/Environment/Roads/Main",(0,0,.02),(extent,24,.1),(.20,.20,.20),"road_main")
        self.secondary_roads=[]
        for i,x in enumerate((-72.,72.)):self.secondary_roads.append(self._cube(f"/World/Environment/Roads/Secondary_{i:02d}",(x,0,.015),(14,extent,.08),(.28,.28,.28),f"road_secondary_{i:02d}"))
        for i,(x,y,color) in enumerate(((-92,-92,(.22,.43,.24)),(92,-92,(.30,.48,.28)),(-92,92,(.20,.39,.31)),(92,92,(.36,.45,.22)))):self._cube(f"/World/Environment/Terrain/Parcel_{i:02d}",(x,y,.005),(105,105,.025),color,f"terrain_parcel_{i:02d}")
        self.markings=[]
        for i,x in enumerate(range(-round(extent/2)+5,round(extent/2)-4,10)):self.markings.append(self._cube(f"/World/Environment/Roads/Main/Marking_{i:02d}",(x,0,.09),(5,.35,.03),(.9,.9,.9),f"lane_marking_{i:02d}"))
        self.persistent=self._cube("/World/Environment/Buildings/Persistent",(-45,-45,5),(24,20,10),(.52,.48,.43),"building_persistent")
        self.development=self._cube("/World/Environment/Buildings/Development",(52,48,6),(28,24,12),(.48,.45,.42),"building_development")
        landmarks=((-105,-105,18,24,8),(-82,-58,22,16,12),(-112,62,20,28,9),(-72,108,26,18,14),(103,-105,28,18,11),(78,-62,18,26,8),(110,65,22,22,15),(75,112,30,17,10),(-28,-110,22,18,9),(30,108,24,20,12),(-112,-8,18,25,8),(108,18,25,18,13),(-35,52,20,16,7),(38,-48,18,22,10))
        for i,(x,y,sx,sy,h) in enumerate(landmarks):self._cube(f"/World/Environment/Buildings/Landmark_{i:02d}",(x,y,h/2),(sx,sy,h),(.38+.025*(i%5),.36,.33),f"building_landmark_{i:02d}")
        rng=np.random.default_rng(cfg.random_seed);self.trees=[]
        half=extent/2;zones=((-half+12,-18,-half+12,-18),(18,half-12,-half+12,-18),(-half+12,-18,18,half-12),(18,half-12,18,half-12))
        for i in range(cfg.changes.tree_initial_count+cfg.changes.tree_new_count):
            x0,x1,y0,y1=zones[i%4];x=float(rng.uniform(x0,x1));y=float(rng.uniform(y0,y1));sphere=UsdGeom.Sphere.Define(self.stage,f"/World/Environment/Vegetation/Tree_{i:03d}");sphere.CreateRadiusAttr(1.0);sphere.CreateDisplayColorAttr([Gf.Vec3f(.12,.5,.14)]);xf=UsdGeom.Xformable(sphere);xf.AddTranslateOp().Set(Gf.Vec3d(x,y,1.4));scale=xf.AddScaleOp();scale.Set(Gf.Vec3f(1,1,1.4));sphere.GetPrim().CreateAttribute("simlab:semanticId",Sdf.ValueTypeNames.String).Set(f"tree_{i:03d}");self.trees.append((sphere,scale))
        dome=UsdLux.DomeLight.Define(self.stage,"/World/Lighting/Dome");self.light=dome.CreateIntensityAttr(1100.);dome.CreateColorAttr(Gf.Vec3f(1,.96,.9))
        aircraft=UsdGeom.Xform.Define(self.stage,"/World/UAV/FixedWing");aircraft_xf=UsdGeom.Xformable(aircraft);self.aircraft_translate=aircraft_xf.AddTranslateOp();self.aircraft_orient=aircraft_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble);self.camera_translate=self.aircraft_translate
        self._cube("/World/UAV/FixedWing/Fuselage",(0,0,0),(5.8,.8,.75),(.72,.16,.10),"fixed_wing_fuselage");self._cube("/World/UAV/FixedWing/MainWing",(-.3,0,0),(1.2,8.5,.18),(.80,.22,.14),"fixed_wing_main_wing");self._cube("/World/UAV/FixedWing/TailWing",(-2.2,0,.05),(.7,3.1,.14),(.72,.16,.10),"fixed_wing_tail_wing");self._cube("/World/UAV/FixedWing/VerticalTail",(-2.25,0,.55),(.7,.16,1.3),(.72,.16,.10),"fixed_wing_vertical_tail")
        camera=UsdGeom.Camera.Define(self.stage,"/World/UAV/FixedWing/Camera");camera.CreateFocalLengthAttr(cfg.camera.focal_length_mm);camera.CreateHorizontalApertureAttr(36.);camera.CreateClippingRangeAttr(Gf.Vec2f(.1,1000));UsdGeom.Xformable(camera).AddTranslateOp().Set(Gf.Vec3d(0,0,-.55));camera.GetPrim().CreateAttribute("simlab:sensorId",Sdf.ValueTypeNames.String).Set("fixed_wing_nadir")
        observer=UsdGeom.Camera.Define(self.stage,"/World/ObserverCamera");observer.CreateFocalLengthAttr(32.);observer.CreateHorizontalApertureAttr(36.);observer.CreateClippingRangeAttr(Gf.Vec2f(.1,2000));observer_xf=UsdGeom.Xformable(observer);observer_position=Gf.Vec3d(0,-230,185);observer_xf.AddTranslateOp().Set(observer_position);aim=Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(0,0,20)-observer_position);observer_xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(aim.GetQuat())
        first=FixedWingTrajectoryManager(cfg.camera).poses()[0];self.set_aircraft_pose(first.x_m,first.y_m,first.z_m,first.yaw_deg)
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath("/World"))

    def _cube(self,path,center,size,color,semantic):
        from pxr import Sdf
        cube=self.UsdGeom.Cube.Define(self.stage,path);cube.CreateSizeAttr(2.);cube.CreateDisplayColorAttr([self.Gf.Vec3f(*color)]);xf=self.UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(self.Gf.Vec3d(*center));scale=xf.AddScaleOp();scale.Set(self.Gf.Vec3f(*(v/2 for v in size)));cube.GetPrim().CreateAttribute("simlab:semanticId",Sdf.ValueTypeNames.String).Set(semantic);return cube

    @staticmethod
    def _color(prim,color):prim.GetDisplayColorAttr().Set([prim.GetDisplayColorAttr().Get()[0].__class__(*color)])

    def apply(self,state):
        from pxr import Gf,UsdGeom
        ground={"spring":(.28,.52,.30),"summer":(.18,.44,.20),"autumn":(.48,.38,.18),"winter":(.72,.75,.76)}[state.season]
        leaf={"spring":(.2,.68,.25),"summer":(.1,.48,.15),"autumn":(.75,.36,.08),"winter":(.38,.4,.38)}[state.season]
        self.ground.GetDisplayColorAttr().Set([Gf.Vec3f(*ground)]);road=float(.20+.22*state.road_appearance_change_score);self.road.GetDisplayColorAttr().Set([Gf.Vec3f(road,road,road)])
        road_scale=self.UsdGeom.Xformable(self.road).GetOrderedXformOps()[1];road_scale.Set(Gf.Vec3f(self.cfg.map.extent_m/2,12*state.road_width_scale,.05))
        for marking in self.markings:marking.GetDisplayColorAttr().Set([Gf.Vec3f(*([.9*state.road_marking_visibility]*3))])
        dev_visible=state.building_variant!="EmptyLot";UsdGeom.Imageable(self.development).MakeVisible() if dev_visible else UsdGeom.Imageable(self.development).MakeInvisible()
        color=(.48,.45,.42) if state.building_variant=="Construction" else ((.62,.52,.43) if state.building_variant=="Completed" else (.65,.32,.18));self.development.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        for i,(tree,scale) in enumerate(self.trees):
            (UsdGeom.Imageable(tree).MakeVisible() if i<state.visible_tree_count else UsdGeom.Imageable(tree).MakeInvisible());scale.Set(Gf.Vec3f(state.tree_scale,state.tree_scale,state.tree_scale*1.4));tree.GetDisplayColorAttr().Set([Gf.Vec3f(*leaf)])
        self.light.Set(1100*state.lighting_scale)
        root=self.stage.GetPrimAtPath("/World/GroundTruth");root.SetCustomDataByKey("epoch",state.epoch);root.SetCustomDataByKey("environmentChangeScore",state.environment_change_score);root.SetCustomDataByKey("structuralChangeScore",state.structural_change_score);root.SetCustomDataByKey("appearanceChangeScore",state.appearance_change_score)

    def set_aircraft_pose(self,x,y,z,yaw_deg):
        self.aircraft_translate.Set(self.Gf.Vec3d(float(x),float(y),float(z)));rotation=self.Gf.Rotation(self.Gf.Vec3d(0,0,1),float(yaw_deg));self.aircraft_orient.Set(rotation.GetQuat())


def _hold_validation_gui(app,scene,cfg,states,summary,validation_image):
    """Keep Isaac open with an in-app epoch controller and research summary."""
    import omni.ui as ui
    try:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().set_active_camera("/World/ObserverCamera")
    except Exception:pass
    selected={"epoch":cfg.epochs[0]};flight_poses=FixedWingTrajectoryManager(cfg.camera).poses();window=ui.Window("Temporal Environment Validation",width=760,height=760)
    with window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8,height=0):
                ui.Label("Accelerated Temporal Environment Validation",height=28,style={"font_size":20})
                status=ui.Label("Select an epoch to inspect the mutable USD state.",height=48,word_wrap=True)
                with ui.HStack(height=34,spacing=5):
                    for epoch in cfg.epochs:
                        def select(target=epoch):
                            selected["epoch"]=target;state=states.create(target);scene.apply(state);pose=flight_poses[0];scene.set_aircraft_pose(pose.x_m,pose.y_m,pose.z_m,pose.yaw_deg);status.text=f"{target.name} | year={target.year:g} | season={target.season} | weather={target.weather} | C_env={state.environment_change_score:.3f} | structural={state.structural_change_score:.3f} | appearance={state.appearance_change_score:.3f}"
                        ui.Button(epoch.name,clicked_fn=select)
                ui.Separator(height=4)
                ui.Label(f"Isaac matching: success={summary['correct']:.3f}, false-match={summary['false']:.3f}, mean inlier={summary['inlier']:.3f}",height=28)
                ui.Label("Reference (left) vs final-epoch query (right); green=RANSAC inlier, red=outlier",height=32,word_wrap=True)
                ui.Image(str(validation_image),height=520,fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)
                ui.Label("The fixed-wing aircraft continuously replays the configured spiral. Close Isaac Sim to finish.",height=28,word_wrap=True)
    pose_index=0;last_step=time.monotonic();period=1/max(.1,cfg.camera.capture_fps)
    while app.is_running():
        now=time.monotonic()
        if now-last_step>=period:
            pose=flight_poses[pose_index];scene.set_aircraft_pose(pose.x_m,pose.y_m,pose.z_m,pose.yaw_deg);epoch=selected["epoch"];state=states.create(epoch);status.text=f"{epoch.name} | aircraft=({pose.x_m:.1f}, {pose.y_m:.1f}, {pose.z_m:.1f}) m | yaw={pose.yaw_deg:.1f} deg | spiral={100*pose.progress:.0f}% | C_env={state.environment_change_score:.3f}";pose_index=(pose_index+1)%len(flight_poses);last_step=now
        app.update()
    return window


def run(config_path:Path,output:Path,headless:bool=True,hold_gui:bool=False)->Path:
    # SimulationApp must be created before importing omni/pxr modules.
    from simlab.sim.app import fresh_stage,launch
    app=launch(headless=headless)
    try:
        print("[temporal-isaac] building USD scene",flush=True)
        fresh_stage(app);cfg=load_temporal_config(config_path);scene=IsaacTemporalScene(cfg)
        import omni.replicator.core as rep
        product=rep.create.render_product("/World/UAV/FixedWing/Camera",cfg.camera.resolution,name="fixed_wing_nadir");rgb=rep.AnnotatorRegistry.get_annotator("rgb");rgb.attach([product])
        states=EnvironmentStateFactory(cfg,TemporalEventScheduler(cfg.events,cfg.random_seed));root=output/"isaac_epoch_images";layers=output/"isaac_epoch_usd_layers";root.mkdir(parents=True,exist_ok=True);layers.mkdir(parents=True,exist_ok=True);captured={}
        flight_poses=FixedWingTrajectoryManager(cfg.camera).poses()
        for epoch in cfg.epochs:
            print(f"[temporal-isaac] applying {epoch.name}",flush=True)
            state=states.create(epoch);scene.apply(state);epoch_dir=root/epoch.name;epoch_dir.mkdir(exist_ok=True)
            captured[epoch.name]=[]
            for pose in flight_poses:
                index=pose.index
                print(f"[temporal-isaac] rendering {epoch.name} pose {index}",flush=True)
                scene.set_aircraft_pose(pose.x_m,pose.y_m,pose.z_m,pose.yaw_deg)
                # Replicator owns the render-product graph; an orchestrator step
                # is required to populate annotators in headless mode. Plain Kit
                # updates alone can leave the RGB buffer empty in Isaac Sim 5.1.
                rep.orchestrator.step(rt_subframes=4)
                frame=np.asarray(rgb.get_data())
                if frame.size==0:raise RuntimeError(f"Isaac RGB annotator returned no data for {epoch.name}")
                if frame.shape[-1]==4:frame=frame[...,:3]
                bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR);cv2.imwrite(str(epoch_dir/f"pose_{index:03d}.png"),bgr);captured[epoch.name].append(bgr)
            scene.stage.GetRootLayer().Export(str(layers/f"{epoch.name}.usda"))
            video_path=output/"isaac_epoch_videos";video_path.mkdir(exist_ok=True);height,width=captured[epoch.name][0].shape[:2];writer=cv2.VideoWriter(str(video_path/f"{epoch.name}_spiral.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),cfg.camera.capture_fps,(width,height))
            for image in captured[epoch.name]:writer.write(image)
            writer.release()
        rgb.detach([product.path])
        matcher=TemporalImageMatcher();reference=captured[cfg.epochs[0].name];matching_rows=[];views=output/"isaac_validation_views";views.mkdir(exist_ok=True)
        for epoch in cfg.epochs[1:]:
            state=states.create(epoch)
            for index,image in enumerate(captured[epoch.name]):
                selected,metrics=matcher.retrieve(image,reference);matching_rows.append({"epoch":epoch.name,"virtual_year":epoch.year,"trajectory_index":index,"selected_reference_index":selected,"is_correct_place_match":selected==index,"false_place_match":selected>=0 and selected!=index,"environment_change_score":state.environment_change_score,"structural_change_score":state.structural_change_score,"appearance_change_score":state.appearance_change_score,**metrics});ref=reference[index if selected<0 else selected];visual=_match_visualization(ref,image,f"Year0 pose {index if selected<0 else selected}  |  {epoch.name} pose {index}  |  inlier ratio {float(metrics['inlier_ratio']):.3f}");cv2.imwrite(str(views/f"{epoch.name}_pose_{index:03d}.png"),visual)
        with (output/"isaac_matching_results.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(matching_rows[0]));writer.writeheader();writer.writerows(matching_rows)
        correct=float(np.mean([row["is_correct_place_match"] for row in matching_rows]));false=float(np.mean([row["false_place_match"] for row in matching_rows]));mean_inlier=float(np.mean([row["inlier_ratio"] for row in matching_rows]))
        with (output/"isaac_summary_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=["metric","value"]);writer.writeheader();writer.writerows([{"metric":"correct_place_match_rate","value":correct},{"metric":"false_match_rate","value":false},{"metric":"mean_inlier_ratio","value":mean_inlier}])
        manifest={"isaac_sim_version":version("isaacsim"),"aircraft_type":"procedural_fixed_wing","trajectory_mode":cfg.camera.trajectory_mode,"map_extent_m":cfg.map.extent_m,"epoch_count":len(cfg.epochs),"camera_pose_count":len(flight_poses),"rgb_image_count":len(cfg.epochs)*len(flight_poses),"video_count":len(cfg.epochs),"usd_layer_count":len(cfg.epochs),"matching_query_count":len(matching_rows),"status":"complete"}
        (output/"isaac_validation.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        print(f"[temporal-isaac] wrote {output}",flush=True)
        if hold_gui and not headless:
            print("[temporal-isaac] validation GUI ready; close Isaac Sim to exit",flush=True)
            final_view=views/f"{cfg.epochs[-1].name}_pose_000.png"
            _hold_validation_gui(app,scene,cfg,states,{"correct":correct,"false":false,"inlier":mean_inlier},final_view)
        return output
    except BaseException as error:
        print(f"[temporal-isaac] FAILED: {type(error).__name__}: {error!r}",flush=True)
        raise
    finally:app.close()


def main()->int:
    parser=argparse.ArgumentParser(description="Validate temporal epochs in Isaac Sim 5.1")
    parser.add_argument("--config",default=str(DEFAULT_TEMPORAL_CONFIG));parser.add_argument("--output",required=True);parser.add_argument("--gui",action="store_true");parser.add_argument("--hold",action="store_true",help="keep Isaac open with the validation panel after capture");args=parser.parse_args()
    print(run(Path(args.config),Path(args.output),not args.gui,args.hold));return 0


if __name__=="__main__":raise SystemExit(main())
