from __future__ import annotations
"""YOLOv8/YOLO11-compatible mine vision layer with optional tracking.

Safety policy: only explicitly mapped classes from the loaded model can create
vision events. A generic COCO model cannot infer missing PPE or falls from the
absence of a class.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib, tempfile, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False

PERSON_KEYS={"person","worker","miner","human"}
EVENT_ALIASES={
    "no_helmet":"NO HELMET","helmet_missing":"NO HELMET","without_helmet":"NO HELMET","no_hardhat":"NO HELMET","no_hard_hat":"NO HELMET",
    "no_vest":"NO SAFETY VEST","vest_missing":"NO SAFETY VEST","without_vest":"NO SAFETY VEST","no_safety_vest":"NO SAFETY VEST",
    "fall":"FALL","fallen":"FALLEN PERSON","fallen_person":"FALLEN PERSON","person_down":"PERSON DOWN","worker_down":"WORKER DOWN",
    "collapse":"COLLAPSE","smoke":"SMOKE","fire":"FIRE","obstacle":"OBSTRUCTION","blocked_exit":"BLOCKED EXIT","unsafe_person":"UNSAFE PERSON",
}
EVENT_WEIGHTS={"FALL":95,"FALLEN PERSON":95,"PERSON DOWN":95,"WORKER DOWN":95,"FIRE":90,"SMOKE":88,"COLLAPSE":95,"BLOCKED EXIT":70,"OBSTRUCTION":65,"NO HELMET":55,"NO SAFETY VEST":45,"UNSAFE PERSON":60}
CRITICAL_EVENTS={"FALL","FALLEN PERSON","PERSON DOWN","WORKER DOWN","FIRE","SMOKE","COLLAPSE"}

@dataclass(frozen=True)
class VisionDetection:
    label:str; confidence:float; box:Tuple[int,int,int,int]; class_id:int; track_id:Optional[int]=None
    def to_dict(self): return asdict(self)
@dataclass(frozen=True)
class VisionEvent:
    event:str; confidence:float; source_class:str; severity:float; track_id:Optional[int]=None
    def to_dict(self): return asdict(self)
@dataclass
class VisionResult:
    detections:List[VisionDetection]=field(default_factory=list); events:List[VisionEvent]=field(default_factory=list)
    person_count:int=0; vision_score:float=0; model_status:str="NOT RUN"; model_name:str="YOLOv8n"; model_source:str="BASELINE"
    model_checksum:Optional[str]=None; processed:Optional[Image.Image]=None; note:str=""; inference_ms:Optional[float]=None
    tracking_enabled:bool=False
    @property
    def unsafe_events(self): return [e.event for e in self.events]
    @property
    def has_actionable_event(self): return bool(self.events)
    @property
    def has_critical_event(self): return any(e.event in CRITICAL_EVENTS for e in self.events)
    @property
    def fall_detected(self): return any(e.event in {"FALL","FALLEN PERSON","PERSON DOWN","WORKER DOWN"} for e in self.events)
    def to_dict(self):
        return {"model_name":self.model_name,"model_source":self.model_source,"model_checksum":self.model_checksum,"model_status":self.model_status,
                "person_count":self.person_count,"vision_score":self.vision_score,"inference_ms":self.inference_ms,"tracking_enabled":self.tracking_enabled,
                "events":[e.to_dict() for e in self.events],"detections":[d.to_dict() for d in self.detections],"note":self.note}

def normalize_label(label:str)->str: return label.strip().lower().replace("-","_").replace(" ","_")
def _event_for_label(label): return EVENT_ALIASES.get(normalize_label(label))
def _checksum(path):
    try:
        p=Path(path)
        if not p.exists(): return None
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        return h.hexdigest()[:16]
    except Exception: return None

def _annotate(img,detections,events):
    out=img.convert("RGB").copy(); draw=ImageDraw.Draw(out); font=ImageFont.load_default()
    alert_keys={(normalize_label(e.source_class),e.track_id) for e in events}
    for d in detections:
        x1,y1,x2,y2=d.box; alert=(normalize_label(d.label),d.track_id) in alert_keys or any(normalize_label(e.source_class)==normalize_label(d.label) for e in events)
        outline=(255,78,78) if alert else (70,220,150)
        draw.rectangle((x1,y1,x2,y2),outline=outline,width=4 if alert else 2)
        suffix=f" ID:{d.track_id}" if d.track_id is not None else ""
        text=f"{d.label} {d.confidence:.0%}{suffix}"; ty=max(0,y1-16); tw=min(out.width,x1+max(100,len(text)*7))
        draw.rectangle((x1,ty,tw,y1),fill=(85,15,15) if alert else (5,30,20)); draw.text((x1+3,ty+3),text,fill=(255,240,240) if alert else (225,255,238),font=font)
    if events:
        banner="VISION ALERT // "+" | ".join(e.event for e in events[:4]); draw.rectangle((0,0,out.width,32),fill=(85,15,15)); draw.text((8,10),banner[:180],fill=(255,240,240),font=font)
    return out

class YOLOVision:
    def __init__(self,model_path="yolov8n.pt",confidence=.35,tracking=True):
        self.model_path=model_path; self.confidence=float(confidence); self.tracking=bool(tracking); self.model=None; self.status="NOT LOADED"; self.model_name=Path(model_path).name; self.checksum=_checksum(model_path)
        self.model_source="CUSTOM" if Path(model_path).suffix.lower()==".pt" and self.model_name!="yolov8n.pt" else "BASELINE"
    def load(self):
        if not YOLO_AVAILABLE: self.status="ULTRALYTICS NOT INSTALLED"; return self.status
        try:
            self.model=YOLO(self.model_path); self.model_name=Path(self.model_path).name; self.checksum=_checksum(self.model_path)
            names=getattr(self.model,"names",{}); self.status="READY"; return f"READY · {len(names)} CLASSES · {self.model_source}"
        except Exception as exc:
            self.model=None; self.status=f"MODEL LOAD FAILED · {str(exc)[:120]}"; return self.status
    def predict(self,image):
        if self.model is None and self.load() and self.model is None:
            return VisionResult(model_status=self.status,model_name=self.model_name,model_source=self.model_source,model_checksum=self.checksum,note="Install ultralytics and ensure model weights are available.")
        try:
            started=time.perf_counter()
            kwargs=dict(source=image,conf=self.confidence,verbose=False)
            if self.tracking: kwargs.update(persist=True,tracker="bytetrack.yaml")
            try: results=self.model.track(**kwargs) if self.tracking else self.model.predict(**kwargs)
            except Exception:
                # Some custom environments/versions may not ship the tracker config; detection remains valid.
                results=self.model.predict(source=image,conf=self.confidence,verbose=False); self.tracking=False
            elapsed=(time.perf_counter()-started)*1000; result=results[0]; names=getattr(result,"names",getattr(self.model,"names",{})); boxes=getattr(result,"boxes",None)
            detections=[]
            if boxes is not None:
                xyxy=boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy(); classes=boxes.cls.cpu().numpy().astype(int)
                ids=boxes.id.cpu().numpy().astype(int) if getattr(boxes,"id",None) is not None else [None]*len(xyxy)
                for box,conf,cls_id,tid in zip(xyxy,confs,classes,ids):
                    label=names.get(int(cls_id),str(cls_id)) if isinstance(names,dict) else str(cls_id)
                    detections.append(VisionDetection(str(label),float(conf),tuple(int(v) for v in box),int(cls_id),None if tid is None else int(tid)))
            persons=[d for d in detections if normalize_label(d.label) in PERSON_KEYS]
            events=[]
            for d in detections:
                event=_event_for_label(d.label)
                if not event: continue
                existing=next((e for e in events if e.event==event and e.track_id==d.track_id),None)
                if existing is None: events.append(VisionEvent(event,d.confidence,d.label,float(EVENT_WEIGHTS[event]),d.track_id))
                elif d.confidence>existing.confidence: events[events.index(existing)]=VisionEvent(event,d.confidence,d.label,float(EVENT_WEIGHTS[event]),d.track_id)
            # Aggregate unique event evidence; confidence-weighted severity, bounded.
            score=min(100.0,max((e.severity*e.confidence for e in events),default=0.0))
            note=("Custom mine-trained model active; mapped safety classes are fused." if self.model_source=="CUSTOM" else
                  "Baseline YOLOv8n is generic COCO. Person detection is valid; missing PPE/fall/fire classes are NOT inferred. Upload a mine-trained model for safety events.")
            return VisionResult(detections,events,len(persons),round(score,1),f"READY · {len(detections)} DETECTIONS",self.model_name,self.model_source,self.checksum,_annotate(image,detections,events),note,round(elapsed,1),self.tracking)
        except Exception as exc:
            return VisionResult(model_status=f"INFERENCE FAILED · {str(exc)[:120]}",model_name=self.model_name,model_source=self.model_source,model_checksum=self.checksum,note="Vision failed; environmental safety logic remains independent.")

def save_uploaded_model(uploaded_file):
    if uploaded_file is None:return None
    payload=uploaded_file.getbuffer(); digest=hashlib.sha256(payload).hexdigest()[:16]; path=Path(tempfile.gettempdir())/f"minesafe_yolo_{digest}.pt"; path.write_bytes(payload); return str(path)

def analyze_structure(img):
    gray=img.convert("L").resize((640,max(1,int(img.height*640/max(1,img.width))))); edges=gray.filter(ImageFilter.FIND_EDGES); arr=np.asarray(edges,dtype=float); density=float(np.mean(arr>45)); attention=min(100,max(0,density*180))
    return VisionResult(model_status="TEXTURE ANALYSIS",model_name="Edge/texture baseline",model_source="NON-AI",vision_score=0,processed=edges,note=f"Informational texture diagnostic. Edge density={density:.3f}; index={attention:.1f}. Not a crack detector and never fused into safety score.")
