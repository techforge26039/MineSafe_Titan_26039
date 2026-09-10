"""Validate a YOLO detection dataset before training.
Checks YAML paths, image/label pairing, class IDs and normalized box ranges.
"""
from pathlib import Path
import argparse, yaml
from PIL import Image

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='dataset/data.yaml'); args=ap.parse_args()
    p=Path(args.data)
    if not p.exists(): raise SystemExit(f'Missing {p}')
    d=yaml.safe_load(p.read_text())
    names=d.get('names',[]); nc=int(d.get('nc',len(names)))
    if not names or nc != len(names): raise SystemExit('Invalid names/nc in data.yaml')
    root=Path(d.get('path',p.parent))
    if not root.is_absolute(): root=(p.parent/root).resolve()
    errors=[]; images=0; labels=0
    for split in ('train','val'):
        raw=d.get(split)
        if not raw: continue
        sp=Path(raw); sp=(root/sp).resolve() if not sp.is_absolute() else sp
        files=[]
        if sp.is_dir(): files=[x for x in sp.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
        elif sp.is_file(): files=[Path(x.strip()) for x in sp.read_text().splitlines() if x.strip()]
        else: errors.append(f'{split}: path missing: {sp}'); continue
        for img in files:
            images+=1; lp=img.with_suffix('.txt')
            if not lp.exists(): errors.append(f'{split}: missing label for {img}'); continue
            labels+=1
            try: Image.open(img).verify()
            except Exception: errors.append(f'{split}: unreadable image {img}'); continue
            for line_no,line in enumerate(lp.read_text().splitlines(),1):
                if not line.strip(): continue
                parts=line.split()
                if len(parts)!=5: errors.append(f'{lp}:{line_no}: expected 5 values'); continue
                try: cls=int(parts[0]); vals=[float(x) for x in parts[1:]]
                except Exception: errors.append(f'{lp}:{line_no}: non-numeric label'); continue
                if not 0<=cls<nc: errors.append(f'{lp}:{line_no}: class {cls} outside 0..{nc-1}')
                if any(not 0<=v<=1 for v in vals): errors.append(f'{lp}:{line_no}: box values must be normalized 0..1')
    print(f'Images: {images} | Labels: {labels} | Classes: {nc}')
    if errors:
        print(f'ERRORS: {len(errors)}')
        for e in errors[:50]: print(' -',e)
        raise SystemExit(1)
    print('YOLO DATASET VALIDATION: PASS')
if __name__=='__main__': main()
