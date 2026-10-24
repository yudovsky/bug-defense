"""
Digitise Shimojo et al. (2023) Sci Rep 13:11898, Fig. 5 (50 W) and Fig. 7 (10 W), from the
vector PDF in ../references/. Rendering at 600 dpi; y axes are calibrated from the detected tick
marks (residual printed per panel); series are found by MATLAB default colours:
experiment = purple (0.494, 0.184, 0.556), with DOP = blue, without DOP = orange.
For each speed: marker centre ('value') and error-bar ends ('bar_lo', 'bar_hi'; +-1 SD, n = 3).

Requires PyMuPDF (pip install pymupdf). Output: ../raw_output/shimojo2023_digitized.json
"""
import json
from pathlib import Path
import numpy as np
import pymupdf
HERE = Path(__file__).resolve().parent
PDF = str(HERE.parent / 'references' / 'Shimojo2023_SciRep_laser_ablation_MC_DOP.pdf')
DPI=600; S=DPI/170
COL={'exp':(126,47,142),'dop':(0,114,189),'nodop':(217,83,25)}
def page_img(pno):
    pix=pymupdf.open(PDF)[pno].get_pixmap(dpi=DPI)
    return np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.h,pix.w,pix.n)[:,:,:3].astype(int)
def axes_box(img,x0,y0,x1,y1):
    c=img[y0:y1,x0:x1]; dark=(c.sum(axis=2)<200)
    cols=dark.sum(axis=0); rows=dark.sum(axis=1)
    L=np.argmax(cols[:len(cols)//2]); R=len(cols)//2+np.argmax(cols[len(cols)//2:])
    T=np.argmax(rows[:len(rows)//2]); B=len(rows)//2+np.argmax(rows[len(rows)//2:])
    return x0+L,y0+T,x0+R,y0+B
def ticks(img,box):
    L,T,R,B=box
    strip=(img[T-2:B+3,L+4:L+10].sum(axis=2)<200).all(axis=1)
    rows=np.nonzero(strip)[0]+T-2
    groups=[]
    for r in rows:
        if groups and r-groups[-1][-1]<=2: groups[-1].append(r)
        else: groups.append([r])
    return [float(np.mean(g)) for g in groups]
def calib(img,box,tick_vals):
    tk=sorted(ticks(img,box),reverse=True)[:len(tick_vals)]  # bottom first; box top may not be a tick
    if len(tk)!=len(tick_vals): raise SystemExit(f'tick count {len(tk)} != {len(tick_vals)} for box {box}: {tk}')
    a,b=np.polyfit(tk,tick_vals,1)
    return lambda r: a*r+b, float(np.max(np.abs(np.polyval([a,b],tk)-tick_vals)))
def series(img,box,f,color,tol=70):
    L,T,R,B=box; c=img[T+3:B-2,L+14:R-2]
    m=(np.abs(c-np.array(color)).sum(axis=2)<tol)
    colcount=m.sum(axis=0); width=R-L; peaks=[]
    for i in np.argsort(colcount)[::-1]:
        if all(abs(i-p)>width*0.12 for p in peaks): peaks.append(i)
        if len(peaks)==3: break
    out=[]
    for p in sorted(peaks):
        rr=np.nonzero(m[:,max(0,p-2):p+3].any(axis=1))[0]
        mc=np.argmax(m[:,max(0,p-25):p+26].sum(axis=1))
        v=lambda r: round(float(f(T+3+r)),3)
        out.append({'value':v(mc),'bar_lo':v(rr.max()),'bar_hi':v(rr.min())})
    return out
def run(pno,panels,speeds):
    img=page_img(pno); res={}
    for name,((a,b,c,d),tv) in panels.items():
        box=axes_box(img,int(a*S),int(b*S),int(c*S),int(d*S))
        f,cal_err=calib(img,box,tv)
        res[name]={'calibration_residual':cal_err,'speeds_mm_s':speeds}
        for k,v in COL.items(): res[name][k]=series(img,box,f,v)
        print(name,'cal resid %.4f'%cal_err)
        for k in COL: print('   %-6s'%k,[(p['value'],p['bar_lo'],p['bar_hi']) for p in res[name][k]])
    return res
out={'fig5_50W':run(5,{'vap_width':((175,140,465,330),list(range(0,7))),'vap_depth':((500,140,790,330),list(range(0,6))),
                      'vap_area':((825,140,1115,330),[0,2,4,6,8,10,12]),
                      'coag_width':((175,390,465,580),[0,5,10,15]),'coag_depth':((500,390,790,580),list(range(0,8)))},[0.5,0.75,1.0]),
     'fig7_10W':run(6,{'coag_width':((165,1055,455,1250),list(range(0,6))),'coag_depth':((490,1055,780,1250),list(range(0,5)))},[1.0,1.25,1.5])}
with open(HERE.parent / 'raw_output' / 'shimojo2023_digitized.json', 'w') as f:
    json.dump(out, f, indent=1)
