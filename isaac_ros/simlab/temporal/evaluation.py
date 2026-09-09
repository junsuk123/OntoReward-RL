"""ORB global retrieval and RANSAC metrics for fixed-T0 temporal matching."""

from __future__ import annotations

import cv2
import numpy as np


class TemporalImageMatcher:
    def __init__(self, nfeatures: int = 1200, ratio: float = .78, ransac_px: float = 4.):
        self.orb=cv2.ORB_create(nfeatures=nfeatures); self.ratio=ratio; self.ransac_px=ransac_px; self.bf=cv2.BFMatcher(cv2.NORM_HAMMING);self._reference_token=None;self._reference_descriptions=[]
    def describe(self,image): return self.orb.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None)
    def compare(self,query,reference) -> dict[str,float|int]:
        qkp,qd=self.describe(query); rkp,rd=self.describe(reference)
        return self._compare_described(qkp,qd,rkp,rd)
    def _compare_described(self,qkp,qd,rkp,rd) -> dict[str,float|int]:
        result={"detected_keypoint_count":len(qkp),"raw_match_count":0,"ransac_inlier_count":0,"inlier_ratio":0.,"reprojection_rmse":float("nan")}
        if qd is None or rd is None or len(qd)<2 or len(rd)<2:return result
        pairs=self.bf.knnMatch(qd,rd,k=2); good=[a for a,b in pairs if a.distance<self.ratio*b.distance]; result["raw_match_count"]=len(good)
        if len(good)<4:return result
        q=np.float32([qkp[m.queryIdx].pt for m in good]); r=np.float32([rkp[m.trainIdx].pt for m in good]); H,mask=cv2.findHomography(q,r,cv2.RANSAC,self.ransac_px)
        if H is None or mask is None:return result
        keep=mask.ravel().astype(bool); result["ransac_inlier_count"]=int(keep.sum()); result["inlier_ratio"]=float(keep.mean())
        if keep.any():
            projected=cv2.perspectiveTransform(q[keep,None,:],H)[:,0]; result["reprojection_rmse"]=float(np.sqrt(np.mean(np.sum((projected-r[keep])**2,axis=1))))
        return result

    def retrieve(self,query,references:list[np.ndarray]) -> tuple[int,dict[str,float|int]]:
        if self._reference_token!=id(references):self._reference_descriptions=[self.describe(ref) for ref in references];self._reference_token=id(references)
        qkp,qd=self.describe(query);results=[self._compare_described(qkp,qd,rkp,rd) for rkp,rd in self._reference_descriptions]
        index=max(range(len(results)),key=lambda i:(results[i]["ransac_inlier_count"],results[i]["raw_match_count"]))
        if results[index]["raw_match_count"]<4:return -1,results[index]
        return index,results[index]
