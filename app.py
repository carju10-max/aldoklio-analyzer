"""
app.py — Aldo&Klio Analyzer (Gradio)
HuggingFace Spaces · ZeroGPU compatible
"""
import os, json, uuid, tempfile, base64, io, functools
from pathlib import Path
from datetime import datetime

import numpy as np
import gradio as gr

# ZeroGPU: HF detecta @spaces.GPU en el archivo principal al arrancar.
# Se define aquí directamente para que el scanner estático lo encuentre.
try:
    import spaces
    @spaces.GPU(duration=1)
    def _gpu_warmup(): pass
except ImportError:
    pass

# ── Library persistence ───────────────────────────────────────────────────────

LIBRARY_DIR  = Path(__file__).parent / "library"
LIBRARY_JSON = LIBRARY_DIR / "index.json"

STEMS_INFO_DISK = {
    'vocals': {'name_es': 'Voces',    'icon': 'V', 'color': '#c8d4e0'},
    'drums':  {'name_es': 'Batería',  'icon': 'D', 'color': '#a0b0c0'},
    'bass':   {'name_es': 'Bajo',     'icon': 'B', 'color': '#8090a0'},
    'other':  {'name_es': 'Otro',     'icon': 'O', 'color': '#687888'},
    'piano':  {'name_es': 'Piano',    'icon': 'P', 'color': '#90a8b8'},
    'guitar': {'name_es': 'Guitarra', 'icon': 'G', 'color': '#b0c0cc'},
}

DEFAULT_STEMS = ['vocals', 'drums', 'bass', 'other']
STEM_CHOICES  = [
    ('Voces',    'vocals'),
    ('Batería',  'drums'),
    ('Bajo',     'bass'),
    ('Otro',     'other'),
    ('Guitarra', 'guitar'),
    ('Piano',    'piano'),
]


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, dict):         return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):return [_to_serializable(i) for i in obj]
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    return obj


def _save_item(item: dict):
    import soundfile as sf
    item_dir = LIBRARY_DIR / item['id']
    item_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in item.items() if k not in ('results', 'stems', 'audio_bytes')}
    if item.get('results'):
        meta['results'] = _to_serializable(item['results'])
    (item_dir / 'meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    if item.get('audio_bytes'):
        (item_dir / 'audio.wav').write_bytes(item['audio_bytes'])
    if item.get('stems'):
        stems_dir = item_dir / 'stems'
        stems_dir.mkdir(exist_ok=True)
        for key, stem in item['stems'].items():
            wav_path = stems_dir / f"{key}.wav"
            if not wav_path.exists():
                wav_path.write_bytes(stem['wav_bytes'])


def _load_item_from_disk(item_id: str) -> dict | None:
    item_dir  = LIBRARY_DIR / item_id
    meta_path = item_dir / 'meta.json'
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding='utf-8'))
    audio_path = item_dir / 'audio.wav'
    data['audio_bytes'] = audio_path.read_bytes() if audio_path.exists() else None
    stems_dir = item_dir / 'stems'
    if stems_dir.exists():
        try:
            import soundfile as sf_
            stems = {}
            for wav_file in sorted(stems_dir.glob('*.wav')):
                key  = wav_file.stem
                info = STEMS_INFO_DISK.get(key)
                if not info:
                    continue
                audio_np, sr = sf_.read(str(wav_file), always_2d=True)
                stems[key] = {
                    'name_es':    info['name_es'],
                    'icon':       info['icon'],
                    'color':      info['color'],
                    'audio_mono': audio_np.mean(axis=1),
                    'sr':         sr,
                    'wav_bytes':  wav_file.read_bytes(),
                }
            data['stems'] = stems if stems else None
        except Exception:
            data['stems'] = None
    else:
        data['stems'] = None
    return data


def _load_library() -> list:
    LIBRARY_DIR.mkdir(exist_ok=True)
    if not LIBRARY_JSON.exists():
        return []
    try:
        entries = json.loads(LIBRARY_JSON.read_text(encoding='utf-8'))
        for e in entries:
            e.setdefault('results', None)
            e.setdefault('stems', None)
            e.setdefault('audio_bytes', None)
        return entries
    except Exception:
        return []


def _save_library_index(items: list):
    LIBRARY_DIR.mkdir(exist_ok=True)
    safe = ('id', 'filename', 'size_mb', 'ext', 'date_added',
            'bpm', 'key', 'key_en', 'duration', 'stems_plan', 'model_name')
    index = [{k: v for k, v in it.items() if k in safe} for it in items]
    LIBRARY_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')


def _delete_item(item_id: str):
    import shutil
    item_dir = LIBRARY_DIR / item_id
    if item_dir.exists():
        shutil.rmtree(item_dir, ignore_errors=True)


# ── Audio compression ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _compress_to_mp3(wav_bytes: bytes, bitrate: str = '96k') -> tuple:
    try:
        from pydub import AudioSegment
        from audio_analysis import MusicAnalyzer
        ffbin = MusicAnalyzer._find_ffmpeg()
        if ffbin:
            import pydub as _pd
            _pd.AudioSegment.converter = ffbin
            _pd.AudioSegment.ffmpeg    = ffbin
            seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            out = io.BytesIO()
            seg.export(out, format='mp3', bitrate=bitrate)
            return out.getvalue(), 'audio/mpeg'
    except Exception:
        pass
    return wav_bytes, 'audio/wav'


# ── Player HTML builder ───────────────────────────────────────────────────────

def build_player_html(results: dict, audio_bytes: bytes, stems: dict | None = None) -> str:
    chords   = results['chords']['chord_changes']
    duration = float(results['duration'])
    bpm      = float(results['tempo']['bpm'])
    beat_offset = float(results['tempo'].get('beat_offset', 0.0))
    key_name = results['key']['key']
    time_sig = int(results['time_signature']['beats_per_measure'])
    pps      = min(80, max(35, 2400 / max(duration, 1)))
    total_w  = int(duration * pps) + 60

    has_stems  = bool(stems)
    stems_list: list[dict] = []

    if has_stems:
        for key, stem in stems.items():
            mono = stem['audio_mono']
            step = max(1, len(mono) // 1600)
            wf   = [round(float(v), 4) for v in mono[::step]]
            data, mime = _compress_to_mp3(stem['wav_bytes'])
            stems_list.append({
                'key':  key,
                'name': stem['name_es'],
                'icon': stem['icon'],
                'color': stem['color'],
                'wf':   wf,
                'b64':  base64.b64encode(data).decode(),
                'mime': mime,
            })
        full_b64 = full_mime = ''
    else:
        data, mime   = _compress_to_mp3(audio_bytes)
        full_b64     = base64.b64encode(data).decode()
        full_mime    = mime

    stems_json  = json.dumps(stems_list)
    chords_json = json.dumps(chords)

    n_rows = (len(stems_list) + 1) if has_stems else 0
    comp_h = 54 + n_rows * 90 + 10 + 68 + 158 + 16

    css = r"""
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#c0c0c0;font-family:'Segoe UI',system-ui,sans-serif;user-select:none;overflow:hidden}
#tp{display:flex;align-items:center;gap:8px;background:#0d0d0d;border-bottom:1px solid #1c1c1c;padding:8px 14px;height:54px}
.tb{background:#161616;border:1px solid #252525;color:#888;border-radius:6px;width:34px;height:34px;font-size:14px;cursor:pointer;transition:background .15s}
.tb:hover{background:#222}.tb:disabled{opacity:.3;cursor:default}
.tb.playing{background:#fff;border-color:#fff;color:#000}
#sk{flex:1;height:4px;background:#1c1c1c;border-radius:2px;cursor:pointer;position:relative}
#sf{height:100%;background:#fff;border-radius:2px;pointer-events:none;width:0%}
#sd{position:absolute;top:-6px;width:16px;height:16px;border-radius:50%;background:#fff;box-shadow:0 0 6px rgba(255,255,255,.4);margin-left:-8px;pointer-events:none;left:0%}
.tt{font-size:.78rem;color:#444;font-variant-numeric:tabular-nums;white-space:nowrap}
#spd{background:#111;border:1px solid #222;color:#666;border-radius:5px;padding:3px 8px;font-size:.75rem;cursor:pointer}
#stlbl{font-size:.68rem;color:#333;min-width:55px}
#klbl{font-size:.78rem;color:#888;font-weight:600;letter-spacing:.5px}
#bd{display:flex}
#cc{width:190px;min-width:190px;border-right:1px solid #181818}
.ctl{height:90px;border-bottom:1px solid #151515;padding:7px 10px;background:#0a0a0a;display:flex;flex-direction:column;justify-content:space-between}
.ct{display:flex;align-items:center;gap:5px}
.bm,.bs{width:21px;height:21px;font-size:.6rem;font-weight:700;border-radius:4px;border:1px solid #222;cursor:pointer;background:transparent;color:#333;transition:all .12s;line-height:1}
.bm.on{background:#fff;border-color:#fff;color:#000}
.bs.on{background:#888;border-color:#888;color:#000}
.ci{font-size:.8rem;font-weight:600;color:#444;min-width:18px;text-align:center}
.cn{font-size:.75rem;font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vr-row{display:flex;align-items:center;gap:5px}
input[type=range].vr{-webkit-appearance:none;flex:1;height:2px;border-radius:1px;outline:none;cursor:pointer;background:#1e1e1e}
input[type=range].vr::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;cursor:pointer;background:var(--tc,#888)}
.vp{font-size:.58rem;color:#333;min-width:26px;text-align:right}
#wc{flex:1;position:relative;overflow:hidden}
#ph{position:absolute;top:0;bottom:0;width:1px;background:#fff;z-index:30;pointer-events:none;left:0;box-shadow:0 0 6px rgba(255,255,255,.5);display:none}
.wr{height:90px;border-bottom:1px solid #131313;cursor:crosshair;position:relative;overflow:hidden;background:#060606}
.wr canvas{position:absolute;top:0;left:0;width:100%;height:100%}
.wr.muted canvas{opacity:.1;filter:grayscale(1)}
.mpills{position:absolute;top:5px;left:6px;display:flex;gap:4px;z-index:10}
.mpill{background:#111;border:1px solid #1e1e1e;color:#444;border-radius:10px;padding:2px 7px;font-size:.6rem;cursor:pointer}
.mpill.sel{background:#1e1e1e;border-color:#333;color:#aaa}
#ps{background:#0a0812;border-top:1px solid #1e1830;padding:10px 16px 8px;display:flex;align-items:flex-start;gap:18px}
#ci{display:flex;flex-direction:column;justify-content:center;min-width:160px;gap:3px}
#cm{font-size:3rem;font-weight:900;color:#2a1f3d;transition:color .25s,text-shadow .25s;line-height:1;letter-spacing:-1px}
#cm.active{color:#c084fc;text-shadow:0 0 30px rgba(192,132,252,.4)}
.ci-lbl{font-size:.53rem;color:#2a1f3d;text-transform:uppercase;letter-spacing:1.2px;margin-top:6px}
#cn{font-size:.95rem;font-weight:700;color:#3d2a5a}
#nn{font-size:.78rem;font-weight:600;color:#7c5fa0;min-height:17px;margin-top:2px}
#nn .en{color:#4a3060;font-size:.68rem}
#pw{flex:1;overflow-x:auto}
canvas#pv{display:block}
#cs{background:#07060e;border-top:1px solid #1a1528;padding:3px 0 0}
#to{overflow-x:auto;overflow-y:hidden;padding:4px 6px 20px;height:68px}
#ti{position:relative;height:44px}
.cb{position:absolute;top:1px;height:42px;border-radius:6px;display:flex;flex-direction:column;justify-content:center;align-items:center;overflow:hidden;border:1px solid rgba(168,85,247,.08);cursor:pointer;transition:border-color .1s,background .1s}
.cb:hover{border-color:rgba(168,85,247,.3)!important;background:rgba(168,85,247,.05)!important}
.cb.active{border:1px solid rgba(192,132,252,.7)!important;background:rgba(124,58,237,.15)!important;box-shadow:0 0 12px rgba(168,85,247,.2);z-index:6}
.cb-n{font-size:11px;font-weight:800;white-space:nowrap;overflow:hidden;max-width:94%;text-overflow:ellipsis;color:#b48dde}
.cb.active .cb-n{color:#e9d5ff}
.cb-t{font-size:8px;color:rgba(168,85,247,.25);margin-top:1px}
#tlph{position:absolute;top:0;bottom:0;width:1px;background:#a855f7;z-index:10;pointer-events:none;left:0;box-shadow:0 0 6px #a855f7}
"""

    js_data = (
        f"const STEMS={stems_json};\n"
        f"const CHORDS={chords_json};\n"
        f"const HAS_STEMS={'true' if has_stems else 'false'};\n"
        f"const FULL_B64='{full_b64}';\n"
        f"const FULL_MIME='{full_mime}';\n"
        f"const BPM={bpm:.3f};\n"
        f"const BEAT_OFFSET={beat_offset:.3f};\n"
        f"const DURATION={duration:.3f};\n"
        f"const TIMESIG={time_sig};\n"
        f"const KEY_NAME='{key_name}';\n"
        f"const PPS={pps:.2f};\n"
        f"const TOTAL_W={total_w};\n"
    )

    js_logic = r"""
const AC=new (window.AudioContext||window.webkitAudioContext)();
const DEST=AC.createGain();DEST.connect(AC.destination);
const gainNodes={},buffers={},volumes={};
const muted=new Set(),soloed=new Set();
let metroGain=null,metroBuf=null,metroMult=1,singleBuf=null;
let isPlaying=false,seekPos=0,ctxStart=0,playRate=1.0;
let activeSrcs=[],metroSrcs=[],rafId=null;

function fmt(s){return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0')}
function getCurTime(){return isPlaying?Math.min(seekPos+(AC.currentTime-ctxStart)*playRate,DURATION):seekPos}
function resumeCtx(){if(AC.state==='suspended')AC.resume()}

function buildGains(){
    if(HAS_STEMS){
        STEMS.forEach(st=>{const g=AC.createGain();g.connect(DEST);gainNodes[st.key]=g;volumes[st.key]=1});
        metroGain=AC.createGain();metroGain.connect(DEST);volumes['metro']=0.7;
    }
}
function applyGains(){
    if(!HAS_STEMS)return;
    const hs=soloed.size>0;
    STEMS.forEach(st=>{const h=!muted.has(st.key)&&(!hs||soloed.has(st.key));gainNodes[st.key].gain.value=h?(volumes[st.key]||1):0});
    metroGain.gain.value=!muted.has('metro')&&(!hs||soloed.has('metro'))?(volumes['metro']||0.7):0;
}
function toggleMute(k){muted.has(k)?muted.delete(k):muted.add(k);document.getElementById('m-'+k)?.classList.toggle('on',muted.has(k));document.getElementById('wr-'+k)?.classList.toggle('muted',muted.has(k));applyGains()}
function toggleSolo(k){soloed.has(k)?soloed.delete(k):soloed.add(k);document.getElementById('s-'+k)?.classList.toggle('on',soloed.has(k));applyGains()}
function setVolume(k,v){volumes[k]=parseFloat(v)/100;document.getElementById('vl-'+k).textContent=v+'%';applyGains()}
function setMetroMult(m){metroMult=m;metroSrcs.forEach(s=>{try{s.stop()}catch(e){}});metroSrcs=[];if(isPlaying)scheduleMetro(getCurTime());document.querySelectorAll('.mpill').forEach(p=>p.classList.toggle('sel',parseFloat(p.dataset.m)===m))}
function setRate(r){playRate=parseFloat(r);if(isPlaying){pauseAll();startAll()}}

async function loadAll(){
    setSt('Cargando audio…');
    async function dec(b64){const bin=atob(b64),u8=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u8[i]=bin.charCodeAt(i);return AC.decodeAudioData(u8.buffer)}
    if(HAS_STEMS){await Promise.all(STEMS.map(async st=>{try{buffers[st.key]=await dec(st.b64)}catch(e){console.error('dec',st.key,e)}}))}
    else if(FULL_B64){try{singleBuf=await dec(FULL_B64)}catch(e){console.error('dec single',e)}}
    setSt('✅ Listo');
    document.getElementById('btn-play').disabled=false;
    setTimeout(()=>{drawAll();buildTL()},180);
}

function makeMetroBuf(){const sr=AC.sampleRate,buf=AC.createBuffer(1,Math.ceil(sr*.055),sr),ch=buf.getChannelData(0);for(let i=0;i<ch.length;i++){const t=i/sr;ch[i]=Math.sin(2*Math.PI*1100*t)*Math.exp(-t*55)}return buf}
function scheduleMetro(from){
    if(!metroBuf)metroBuf=makeMetroBuf();
    const iv=60/(BPM*metroMult);
    const firstBeat=BEAT_OFFSET%iv;
    const relFrom=from-firstBeat;
    const sb=Math.ceil(relFrom/iv),eb=Math.ceil((DURATION-firstBeat)/iv);
    for(let b=sb;b<eb;b++){
        const bt=firstBeat+b*iv;
        if(bt<0||bt>DURATION)continue;
        const st=AC.currentTime+(bt-from)/playRate;
        if(st<AC.currentTime+.01)continue;
        const src=AC.createBufferSource();src.buffer=metroBuf;src.playbackRate.value=playRate;
        const env=AC.createGain();
        const beatInBar=Math.round((bt-BEAT_OFFSET)/iv)%Math.round(TIMESIG*metroMult);
        const isA=beatInBar===0;
        const vol=(volumes['metro']||.7)*(isA?1:.5);
        env.gain.setValueAtTime(vol,st);env.gain.exponentialRampToValueAtTime(.001,st+.07);
        src.connect(env);env.connect(metroGain);src.start(st);metroSrcs.push(src);
    }
}

function startAll(){
    resumeCtx();ctxStart=AC.currentTime;
    if(HAS_STEMS){STEMS.forEach(st=>{if(!buffers[st.key])return;const src=AC.createBufferSource();src.buffer=buffers[st.key];src.playbackRate.value=playRate;src.connect(gainNodes[st.key]);src.start(0,seekPos);activeSrcs.push(src)});scheduleMetro(seekPos)}
    else{if(!singleBuf)return;const src=AC.createBufferSource();src.buffer=singleBuf;src.playbackRate.value=playRate;src.connect(DEST);src.start(0,seekPos);activeSrcs.push(src)}
    isPlaying=true;updTp();rafId=requestAnimationFrame(tick);
}
function pauseAll(){activeSrcs.forEach(s=>{try{s.stop()}catch(e){}});metroSrcs.forEach(s=>{try{s.stop()}catch(e){}});activeSrcs=[];metroSrcs=[];seekPos=Math.min(getCurTime(),DURATION);isPlaying=false;cancelAnimationFrame(rafId);updTp()}
function stopAll(){pauseAll();seekPos=0;updPH(0);updTime(0)}
function togglePlay(){isPlaying?pauseAll():startAll()}
function seekTo(t){const was=isPlaying;pauseAll();seekPos=Math.max(0,Math.min(t,DURATION));was?startAll():(updPH(seekPos/DURATION),updTime(seekPos))}

const ES_MAP={Sol:7,Si:11,Fa:5,Re:2,Mi:4,La:9,Do:0};
const ES_KEYS=['Sol','Si','Fa','Re','Mi','La','Do'];
const IVS={'M7':[0,4,7,11],'m7':[0,3,7,10],'dim':[0,3,6],'aug':[0,4,8],'sus4':[0,5,7],'sus2':[0,2,7],'7':[0,4,7,10],'m':[0,3,7],'':[0,4,7]};
const ESN=['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si'];
const ENN=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function c2n(name){if(!name||name==='N/C'||name==='—')return[];let root=-1,rest=name;for(const n of ES_KEYS){if(name.startsWith(n)){root=ES_MAP[n];rest=name.slice(n.length);break}}if(root<0)return[];if(rest[0]==='#'){root=(root+1)%12;rest=rest.slice(1)}return(IVS[rest]||IVS['']).map(iv=>(root+iv)%12)}
function cCol(n){if(n.includes('dim'))return{bg:'rgba(255,255,255,.04)',txt:'#666',brd:'rgba(255,255,255,.08)'};if(n.includes('aug'))return{bg:'rgba(255,255,255,.06)',txt:'#777',brd:'rgba(255,255,255,.1)'};if(n.includes('sus'))return{bg:'rgba(255,255,255,.05)',txt:'#666',brd:'rgba(255,255,255,.09)'};if(n.includes('m7')||n.includes('M7'))return{bg:'rgba(255,255,255,.07)',txt:'#888',brd:'rgba(255,255,255,.12)'};if(n.endsWith('7'))return{bg:'rgba(255,255,255,.06)',txt:'#777',brd:'rgba(255,255,255,.1)'};if(n.includes('m'))return{bg:'rgba(255,255,255,.05)',txt:'#666',brd:'rgba(255,255,255,.09)'};return{bg:'rgba(255,255,255,.08)',txt:'#999',brd:'rgba(255,255,255,.14)'}}

function buildTL(){
    const ti=document.getElementById('ti');if(!ti)return;
    ti.style.width=TOTAL_W+'px';
    CHORDS.forEach((ch,i)=>{const nxt=i+1<CHORDS.length?CHORDS[i+1].time:DURATION;const x=ch.time*PPS,w=Math.max(4,(nxt-ch.time)*PPS-2),c=cCol(ch.chord);const d=document.createElement('div');d.className='cb';d.dataset.idx=i;d.style.cssText='left:'+x+'px;width:'+w+'px;background:'+c.bg+';border-color:'+c.brd;if(w>22)d.innerHTML='<span class="cb-n" style="color:'+c.txt+'">'+ch.chord+'</span><span class="cb-t">'+ch.time_fmt+'</span>';d.addEventListener('click',()=>seekTo(ch.time));ti.appendChild(d)});
    const step=DURATION<=60?5:DURATION<=180?10:30;
    for(let t=0;t<=DURATION;t+=step){const tk=document.createElement('div');tk.style.cssText='position:absolute;bottom:-16px;left:'+(t*PPS)+'px;transform:translateX(-50%)';tk.innerHTML='<div style="width:1px;height:4px;background:rgba(255,255,255,.12);margin:0 auto"></div><div style="font-size:7px;color:#37474f;text-align:center;white-space:nowrap">'+fmt(t)+'</div>';ti.appendChild(tk)}
}

const KM=[{b:false,w:0},{b:true,w:0},{b:false,w:1},{b:true,w:1},{b:false,w:2},{b:false,w:3},{b:true,w:3},{b:false,w:4},{b:true,w:4},{b:false,w:5},{b:true,w:5},{b:false,w:6}];
const OCT=2,WW=32,WH=86,BW=19,BH=54;
function drawPiano(hi){
    const cv=document.getElementById('pv');if(!cv)return;
    const ctx=cv.getContext('2d');cv.width=7*OCT*WW+1;cv.height=WH+18;
    function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath()}
    for(let o=0;o<OCT;o++)for(let s=0;s<12;s++){const k=KM[s];if(k.b)continue;const x=(o*7+k.w)*WW,on=hi.includes(s);ctx.fillStyle=on?'#e8e8e8':'#1e1e1e';ctx.strokeStyle=on?'#aaa':'#2a2a2a';ctx.lineWidth=1;rr(x+.5,.5,WW-1,WH-1,3);ctx.fill();ctx.stroke();ctx.fillStyle=on?'#000':'#3a3a3a';ctx.font='8px system-ui';ctx.textAlign='center';ctx.fillText(ENN[s],x+WW/2,WH-6);if(on){ctx.fillStyle='#000';ctx.font='bold 8px system-ui';ctx.fillText(ESN[s],x+WW/2,WH-16)}}
    for(let o=0;o<OCT;o++)for(let s=0;s<12;s++){const k=KM[s];if(!k.b)continue;const x=(o*7+k.w)*WW+WW-BW/2,on=hi.includes(s);ctx.fillStyle=on?'#cccccc':'#0a0a0a';ctx.strokeStyle=on?'#888':'#1a1a1a';ctx.lineWidth=1;rr(x+.5,.5,BW-1,BH-1,3);ctx.fill();ctx.stroke();if(on){ctx.fillStyle='#000';ctx.font='bold 8px system-ui';ctx.textAlign='center';ctx.fillText(ENN[s],x+BW/2,BH-8)}}
    ctx.strokeStyle='rgba(255,255,255,.06)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(7*WW,0);ctx.lineTo(7*WW,WH);ctx.stroke();
}

function drawWF(id,wf,color){const cv=document.getElementById(id);if(!cv)return;const dpr=window.devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=cv.offsetHeight*dpr;const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);const W=cv.offsetWidth,H=cv.offsetHeight,mid=H/2;ctx.fillStyle='#060606';ctx.fillRect(0,0,W,H);const n=wf.length,bw=Math.max(1,W/n),mx=Math.max(...wf.map(Math.abs))||1;const hx=color.replace('#',''),r=parseInt(hx.slice(0,2),16),g=parseInt(hx.slice(2,4),16),b=parseInt(hx.slice(4,6),16);ctx.fillStyle=`rgba(${r},${g},${b},0.75)`;for(let i=0;i<n;i++){const a=(Math.abs(wf[i])/mx)*mid*.82;ctx.fillRect(i*bw,mid-a,bw*.7,a*2)}}
function drawMetroWF(id){const cv=document.getElementById(id);if(!cv)return;const dpr=window.devicePixelRatio||1;cv.width=cv.offsetWidth*dpr;cv.height=cv.offsetHeight*dpr;const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);const W=cv.offsetWidth,H=cv.offsetHeight;ctx.fillStyle='#060606';ctx.fillRect(0,0,W,H);const iv=60/BPM,tot=Math.floor(DURATION/iv);for(let b=0;b<tot;b++){const x=(b*iv/DURATION)*W,isA=b%TIMESIG===0,bH=isA?H*.75:H*.38;ctx.fillStyle=isA?'rgba(255,255,255,.55)':'rgba(255,255,255,.18)';ctx.fillRect(x,(H-bH)/2,Math.max(2,W/tot*.45),bH)}}
function drawAll(){if(HAS_STEMS){STEMS.forEach(st=>drawWF('wf-'+st.key,st.wf,st.color));drawMetroWF('wf-metro')}drawPiano([])}

function updTp(){const btn=document.getElementById('btn-play');if(btn){btn.textContent=isPlaying?'⏸':'▶';btn.classList.toggle('playing',isPlaying)}}
function setSt(m){const e=document.getElementById('stlbl');if(e)e.textContent=m}
function updTime(t){const e=document.getElementById('tc');if(e)e.textContent=fmt(t)}
function updPH(frac){
    const ph=document.getElementById('ph'),wc=document.getElementById('wc');
    if(ph&&wc){ph.style.left=(frac*wc.clientWidth)+'px';ph.style.display='block'}
    const sf=document.getElementById('sf'),sd=document.getElementById('sd');
    if(sf)sf.style.width=(frac*100)+'%';if(sd)sd.style.left=(frac*100)+'%';
    const tp=document.getElementById('tlph'),ti=document.getElementById('ti');
    if(tp&&ti){tp.style.left=(frac*TOTAL_W)+'px'}
    const to=document.getElementById('to');
    if(to){const x=frac*TOTAL_W;to.scrollLeft=Math.max(0,x-to.clientWidth/2)}
}
let lastCI=-1;
function updChord(t){
    let idx=-1;for(let i=CHORDS.length-1;i>=0;i--){if(t>=CHORDS[i].time){idx=i;break}}
    if(idx===lastCI)return;lastCI=idx;
    const cmEl=document.getElementById('cm'),cnEl=document.getElementById('cn'),nnEl=document.getElementById('nn');
    if(idx>=0){const ch=CHORDS[idx].chord;if(cmEl){cmEl.textContent=ch;cmEl.classList.toggle('active',ch!=='N/C'&&ch!=='—')}const nx=idx+1<CHORDS.length?CHORDS[idx+1].chord:'—';if(cnEl){cnEl.textContent=nx}const notes=c2n(ch);drawPiano(notes);if(nnEl){if(notes.length){const es=notes.map(n=>ESN[n]),en=notes.map(n=>ENN[n]);nnEl.innerHTML=es.join(' · ')+' <span class="en">('+en.join(' - ')+')</span>'}else nnEl.textContent=''}document.querySelectorAll('.cb').forEach((b,i)=>b.classList.toggle('active',i===idx))}
    else{if(cmEl){cmEl.textContent='—';cmEl.classList.remove('active')}if(cnEl)cnEl.textContent='—';if(nnEl)nnEl.textContent='';drawPiano([])}
}
function tick(){const t=getCurTime();if(t>=DURATION){stopAll();return}updPH(t/DURATION);updTime(t);updChord(t);rafId=requestAnimationFrame(tick)}

document.getElementById('sk').addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*DURATION)});
const wc=document.getElementById('wc');
if(wc)wc.addEventListener('click',e=>{const r=wc.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*DURATION)});
const to=document.getElementById('to');
if(to)to.addEventListener('click',e=>{const ti=document.getElementById('ti');if(!ti)return;const r=ti.getBoundingClientRect();seekTo(Math.max(0,Math.min(1,(e.clientX-r.left)/TOTAL_W))*DURATION)});

buildGains();loadAll();
document.getElementById('tt').textContent=fmt(DURATION);
const kl=document.getElementById('klbl');if(kl)kl.textContent=KEY_NAME;
window.addEventListener('resize',()=>setTimeout(drawAll,100));
"""

    def ctl_row(key, icon, name, color, vol=100):
        return (
            f"<div class='ctl' style='border-left:4px solid {color}'>"
            f"<div class='ct'>"
            f"<button class='bm' id='m-{key}' onclick=\"toggleMute('{key}')\">M</button>"
            f"<button class='bs' id='s-{key}' onclick=\"toggleSolo('{key}')\">S</button>"
            f"<span class='ci'>{icon}</span>"
            f"<span class='cn' style='color:{color}'>{name}</span></div>"
            f"<div class='vr-row'>"
            f"<input type='range' class='vr' id='vol-{key}' style='--tc:{color}' "
            f"min='0' max='200' value='{vol}' oninput=\"setVolume('{key}',this.value)\">"
            f"<span class='vp' id='vl-{key}'>{vol}%</span></div></div>"
        )

    def wf_row(key, extra=""):
        return f"<div class='wr' id='wr-{key}'><canvas id='wf-{key}'></canvas>{extra}</div>"

    mixer_html = ""
    if has_stems:
        ctrl_rows  = "".join(ctl_row(st['key'], st['icon'], st['name'], st['color']) for st in stems_list)
        ctrl_rows += ctl_row('metro', 'M', 'Metrónomo', '#666666', 70)
        wave_rows  = "".join(wf_row(st['key']) for st in stems_list)
        mpills = ("<div class='mpills'>"
                  "<span class='mpill' data-m='0.5' onclick='setMetroMult(0.5)'>0.5×</span>"
                  "<span class='mpill sel' data-m='1' onclick='setMetroMult(1)'>1×</span>"
                  "<span class='mpill' data-m='2' onclick='setMetroMult(2)'>2×</span></div>")
        wave_rows += wf_row('metro', mpills)
        mixer_html = (
            f"<div id='bd'>"
            f"<div id='cc'>{ctrl_rows}</div>"
            f"<div id='wc'><div id='ph'></div>{wave_rows}</div>"
            f"</div>"
        )

    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{css}"
        f"input[type=range].vr::-webkit-slider-thumb{{background:var(--tc,#888)}}"
        f"</style></head><body>"
        f"<div id='tp'>"
        f"<button class='tb' id='btn-play' onclick='togglePlay()' disabled>▶</button>"
        f"<button class='tb' onclick='stopAll()'>⏹</button>"
        f"<span class='tt' id='tc'>0:00</span>"
        f"<div id='sk'><div id='sf'></div><div id='sd'></div></div>"
        f"<span class='tt' id='tt'>--:--</span>"
        f"<select id='spd' onchange='setRate(this.value)'>"
        f"<option value='0.5'>0.5×</option><option value='1' selected>1×</option>"
        f"<option value='1.5'>1.5×</option><option value='2'>2×</option></select>"
        f"<span id='klbl'></span>"
        f"<span id='stlbl'>Cargando…</span>"
        f"</div>"
        f"{mixer_html}"
        f"<div id='cs'><div id='to'><div id='ti'><div id='tlph'></div></div></div></div>"
        f"<div id='ps'>"
        f"<div id='ci'>"
        f"<div id='cm'>—</div>"
        f"<div class='ci-lbl'>SIGUIENTE</div><div id='cn'>—</div>"
        f"<div class='ci-lbl' style='margin-top:4px'>NOTAS</div><div id='nn'></div>"
        f"</div>"
        f"<div id='pw'><canvas id='pv'></canvas></div>"
        f"</div>"
        f"<script>{js_data}{js_logic}</script>"
        f"</body></html>"
    )

    # srcdoc evita el bloqueo de data: URIs por CSP en HF Spaces
    escaped = html.replace('&', '&amp;').replace('"', '&quot;')
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:{comp_h}px;border:none;border-radius:10px;'
        f'background:#0a0a0a" scrolling="no"></iframe>'
    )


# ── Library HTML ──────────────────────────────────────────────────────────────

def build_library_html(library: list) -> str:
    if not library:
        return (
            "<div style='text-align:center;padding:5rem 2rem;"
            "border:1px dashed #141414;border-radius:14px;margin-top:1rem'>"
            "<div style='font-size:2rem;color:#1e1e1e;margin-bottom:14px'>&#9836;</div>"
            "<p style='color:#2e2e2e;font-size:.92rem;margin:0'>Aún no tienes ningún archivo</p>"
            "<p style='color:#1e1e1e;font-size:.78rem;margin:6px 0 0'>"
            "Haz clic en <strong style=\"color:#444\">+ Agregar</strong> para empezar"
            "</p></div>"
        )

    header = (
        "<div style='display:grid;"
        "grid-template-columns:minmax(0,3fr) 110px 60px 90px 80px 80px;"
        "padding:5px 12px;border-bottom:1px solid #141414;"
        "color:#1e1e1e;font-size:.65rem;text-transform:uppercase;letter-spacing:.9px;margin-top:1rem'>"
        "<span>Título</span><span>Fecha</span><span>BPM</span>"
        "<span>Tono</span><span>Duración</span><span></span></div>"
    )

    rows = ""
    for item in library:
        dur  = item.get('duration', 0)
        ds   = f"{int(dur//60)}:{int(dur%60):02d}" if dur else '—'
        ext  = (item.get('ext', '') or '').upper()
        name = item.get('filename', 'Archivo')
        iid  = item['id']
        rows += (
            f"<div style='display:grid;grid-template-columns:minmax(0,3fr) 110px 60px 90px 80px 80px;"
            f"padding:10px 12px;border-bottom:1px solid #0d0d0d;align-items:center'>"
            f"<div style='color:#aaa;font-size:.83rem;font-weight:500;"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{name}"
            f"<span style='color:#252525;font-size:.68rem;margin-left:6px'>{ext}</span></div>"
            f"<div style='color:#303030;font-size:.76rem'>{(item.get('date_added','') or '')[:10]}</div>"
            f"<div style='color:#404040;font-size:.79rem'>{item.get('bpm','—')}</div>"
            f"<div style='color:#404040;font-size:.79rem'>{item.get('key_en', item.get('key','—'))}</div>"
            f"<div style='color:#404040;font-size:.79rem'>{ds}</div>"
            f"<div>"
            f"<button onclick=\"selectItem('{iid}')\" style='"
            f"background:transparent;border:1px solid #2a2a2a;color:#888;border-radius:6px;"
            f"padding:3px 12px;font-size:.72rem;cursor:pointer'>Abrir</button>"
            f"</div></div>"
        )

    script = """
    <script>
    function selectItem(id){
        const tb = document.querySelector('#item-id-box input') || document.querySelector('#item-id-box textarea');
        if(tb){tb.value=id;tb.dispatchEvent(new Event('input',{bubbles:true}));tb.dispatchEvent(new Event('change',{bubbles:true}))}
    }
    </script>"""

    return header + f"<div>{rows}</div>" + script


# ── Log renderer ──────────────────────────────────────────────────────────────

def _render_log(log: list) -> str:
    COLORS = {'ok': '#4ade80', 'info': '#818cf8', 'warn': '#fb923c', 'error': '#f87171'}
    ICONS  = {'ok': '✓', 'info': '·', 'warn': '!', 'error': '✕'}
    rows   = ""
    for e in log:
        c    = COLORS.get(e['level'], '#888')
        icon = ICONS.get(e['level'],  '·')
        msg  = e['msg'].replace('<', '&lt;').replace('>', '&gt;')
        rows += (
            f"<tr>"
            f"<td style='color:#333;font-size:.7rem;padding:3px 8px 3px 0;white-space:nowrap'>+{e['elapsed']}s</td>"
            f"<td style='text-align:center;padding:3px 6px 3px 0'>"
            f"<span style='color:{c};font-weight:700;font-size:.8rem'>{icon}</span></td>"
            f"<td style='color:#555;font-size:.75rem;white-space:nowrap;padding:3px 10px 3px 0;"
            f"font-weight:600'>{e['step']}</td>"
            f"<td style='color:#888;font-size:.72rem;font-family:monospace'>{msg}</td>"
            f"</tr>"
        )
    return (
        f"<div style='background:#060606;border:1px solid #141414;border-radius:8px;"
        f"padding:10px 14px;margin-top:8px'>"
        f"<table style='width:100%;border-collapse:collapse'>{rows}</table></div>"
    )


# ── Gradio CSS ────────────────────────────────────────────────────────────────

GRADIO_CSS = """
footer { display:none !important; }
/* Ocultar cabeceras de tabs */
.tabs > div:first-child button[role="tab"] { display:none !important; }
.tab-nav { height:0 !important; overflow:hidden !important; }

.top-nav {
    display:flex; align-items:center;
    border-bottom:1px solid #1a1a1a;
    padding:0 16px; height:52px; margin-bottom:16px;
}
.top-nav-logo {
    font-size:1.3rem; font-weight:900; letter-spacing:-.5px;
    background:linear-gradient(135deg,#fff 0%,#c084fc 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.top-nav-logo span {
    background:linear-gradient(135deg,#888 0%,#666 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    font-weight:300;
}

#item-id-box { position:fixed; left:-9999px; top:0; width:1px; height:1px; overflow:hidden; opacity:0; }
input[type=checkbox] { accent-color:#c084fc; }
"""


# ── Gradio App ────────────────────────────────────────────────────────────────

def _upd(*visible_flags):
    return [gr.update(visible=v) for v in visible_flags]


ALL_SCREENS = 4  # library, upload, instruments, detail


with gr.Blocks(title="Aldo&Klio Analyzer") as demo:

    library_st = gr.State(_load_library())
    pending_st = gr.State({})

    gr.HTML("<div class='top-nav'><div class='top-nav-logo'>Aldo&amp;Klio<span> Analyzer</span></div></div>")

    # ── Navegación con gr.Tabs (cabeceras ocultas via CSS) ───────────────────
    with gr.Tabs(selected=0, elem_id="screens") as screens:

        # ── TAB 0: LIBRARY ───────────────────────────────────────────────────
        with gr.Tab("", id=0):
            with gr.Row():
                gr.HTML("<h2 style='font-size:1.2rem;font-weight:600;margin:0;padding:4px 0'>Separación de pistas</h2>")
                add_btn = gr.Button("+ Agregar", variant="primary", scale=0, min_width=130)
            lib_html    = gr.HTML(build_library_html(_load_library()))
            item_id_box = gr.Textbox(visible=False, elem_id="item-id-box")

        # ── TAB 1: UPLOAD ────────────────────────────────────────────────────
        with gr.Tab("", id=1):
            back_new_btn = gr.Button("← Volver", size="sm", scale=0)
            gr.HTML("<h2 style='font-size:1.12rem;font-weight:600;margin:12px 0'>Agregar archivo</h2>")

            # ── YouTube download ──────────────────────────────────────────────
            gr.HTML("<div style='font-size:.8rem;color:#555;margin-bottom:6px'>▶ Descargar de YouTube</div>")
            with gr.Row():
                yt_url_in  = gr.Textbox(placeholder="https://www.youtube.com/watch?v=...", show_label=False, scale=4)
                yt_btn     = gr.Button("Descargar", scale=1)
            yt_status = gr.HTML()
            gr.HTML("<div style='border-top:1px solid #1a1a1a;margin:14px 0 10px'></div>")

            file_in = gr.File(
                label="O seleccionar archivo de audio o video",
                file_types=[".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aiff",
                            ".mp4", ".mkv", ".avi", ".mov", ".webm"],
            )
            continue_btn = gr.Button("Continuar →", variant="primary")

        # ── TAB 2: INSTRUMENTS ───────────────────────────────────────────────
        with gr.Tab("", id=2):
            back_instr_btn = gr.Button("← Volver", size="sm", scale=0)
            gr.HTML("<h1 style='font-size:1.7rem;font-weight:800;margin:12px 0 4px'>Separar pistas</h1>")
            instr_file_html = gr.HTML()
            with gr.Row():
                preset_4_btn = gr.Button("Voz · Batería · Bajo · Otro  (4 pistas)", variant="primary")
                preset_2_btn = gr.Button("Voz · Instrumental  (2 pistas)", variant="secondary")
            stem_checks = gr.CheckboxGroup(
                choices=[('Voces','vocals'),('Batería','drums'),('Bajo','bass'),
                         ('Otro','other'),('Guitarra ★6 pistas','guitar'),('Piano ★6 pistas','piano')],
                value=DEFAULT_STEMS, label="Personalizado",
            )
            start_btn    = gr.Button("Empezar separación", variant="primary", size="lg")
            progress_out = gr.HTML()

        # ── TAB 3: DETAIL ────────────────────────────────────────────────────
        with gr.Tab("", id=3):
            with gr.Row():
                back_detail_btn = gr.Button("← Volver", size="sm", scale=0)
                detail_header   = gr.HTML()
            player_out   = gr.HTML()
            download_out = gr.File(label="Descargar pistas (MP3)", file_count="multiple",
                                   interactive=False, visible=False)
            kpi_out      = gr.HTML()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _go(tab_id):
        return gr.update(selected=tab_id)

    def _file_info(file):
        if file is None:
            return {}
        if hasattr(file, 'path'):
            fp   = file.path
            name = getattr(file, 'orig_name', None) or os.path.basename(fp)
            sb   = getattr(file, 'size', None)
            mb   = round((sb/1_048_576) if sb else os.path.getsize(fp)/1_048_576, 2)
        elif hasattr(file, 'name'):
            fp = file.name; name = os.path.basename(fp); mb = round(os.path.getsize(fp)/1_048_576, 2)
        elif isinstance(file, str):
            fp = file; name = os.path.basename(fp); mb = round(os.path.getsize(fp)/1_048_576, 2)
        else:
            return {}
        ext = name.rsplit('.',1)[-1].lower() if '.' in name else ''
        print(f"[file] {name} {mb}MB")
        return {'path': fp, 'name': name, 'ext': ext, 'size_mb': mb}

    def _kpi_html(results):
        d = results.get('duration', 0)
        return (
            "<div style='border-top:1px solid #111;padding-top:14px;"
            "display:flex;gap:24px;flex-wrap:wrap;margin-top:12px'>"
            + "".join(
                f"<div style='text-align:center'>"
                f"<div style='color:#555;font-size:.62rem;text-transform:uppercase;letter-spacing:1px'>{l}</div>"
                f"<div style='font-size:.88rem;font-weight:600'>{v}</div></div>"
                for l,v in [
                    ('Tonalidad', results['key']['key_en']),
                    ('Tempo',     f"{results['tempo']['bpm']} BPM"),
                    ('Compás',    results['time_signature']['time_sig']),
                    ('Duración',  f"{int(d//60)}:{int(d%60):02d}"),
                ]
            ) + "</div>"
        )

    # ── Library navigation ────────────────────────────────────────────────────
    add_btn.click(lambda _: (_go(1), None, {}), inputs=[library_st], outputs=[screens, file_in, pending_st])

    back_new_btn.click(lambda lib: (_go(0), build_library_html(lib)),
                       inputs=[library_st], outputs=[screens, lib_html])

    back_instr_btn.click(lambda _: _go(1), inputs=[library_st], outputs=[screens])

    back_detail_btn.click(lambda lib: (_go(0), build_library_html(lib)),
                          inputs=[library_st], outputs=[screens, lib_html])

    # ── YouTube download ──────────────────────────────────────────────────────
    def download_yt(url):
        if not url or not url.strip():
            return "<p style='color:#f87171;font-size:.8rem'>Pega una URL de YouTube.</p>", {}
        try:
            import yt_dlp
            tmpdir = tempfile.mkdtemp()
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, '%(title)s.%(ext)s'),
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
                'quiet': True, 'no_warnings': True,
                'extractor_args': {'youtube': {'player_client': ['ios', 'tv']}},
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url.strip(), download=True)
                title = info.get('title', 'audio')
            files = [f for f in os.listdir(tmpdir) if f.endswith('.mp3')]
            if not files:
                return "<p style='color:#f87171;font-size:.8rem'>No se pudo descargar el audio.</p>", {}
            fp = os.path.join(tmpdir, files[0])
            mb = round(os.path.getsize(fp) / 1_048_576, 2)
            pending = {'path': fp, 'name': files[0], 'ext': 'mp3', 'size_mb': mb}
            msg = (f"<p style='color:#4ade80;font-size:.8rem'>✓ <b>{title}</b> — {mb} MB · "
                   f"listo para analizar, da <b>Continuar →</b></p>")
            return msg, pending
        except Exception as e:
            return f"<p style='color:#f87171;font-size:.8rem'>Error: {e}</p>", {}

    yt_btn.click(download_yt, inputs=[yt_url_in], outputs=[yt_status, pending_st])

    # ── File upload ───────────────────────────────────────────────────────────
    file_in.upload(_file_info, inputs=[file_in], outputs=[pending_st])
    file_in.change(_file_info, inputs=[file_in], outputs=[pending_st])

    # ── Continue → Instruments ────────────────────────────────────────────────
    def go_instr(pending):
        if not pending or not pending.get('path'):
            return _go(1), ""
        ext  = pending.get('ext','')
        html = (f"<div style='padding:10px 16px;border:1px solid #333;border-radius:8px;margin-bottom:12px'>"
                f"<b>{pending.get('name','')}</b>"
                f"<span style='color:#888;margin-left:8px;font-size:.8rem'>"
                f"{pending.get('size_mb',0):.1f} MB · {ext.upper()}</span></div>")
        return _go(2), html

    continue_btn.click(go_instr, inputs=[pending_st], outputs=[screens, instr_file_html])

    # ── Preset buttons ────────────────────────────────────────────────────────
    preset_4_btn.click(lambda: list(DEFAULT_STEMS), outputs=[stem_checks])
    preset_2_btn.click(lambda: ['vocals','other'],  outputs=[stem_checks])

    # ── Run analysis ──────────────────────────────────────────────────────────
    def run_analysis(pending, sel_stems, library):
        import time as _time
        from audio_analysis import MusicAnalyzer

        if not pending or not pending.get('path'):
            yield _go(2), "<p style='color:red'>Sin archivo.</p>", gr.update(), gr.update(), gr.update(visible=False), gr.update(), library
            return

        t0 = _time.monotonic()
        log = []

        def _log(lv, step, msg):
            log.append({'ts': _time.strftime('%H:%M:%S'),
                        'elapsed': round(_time.monotonic()-t0,2),
                        'level': lv, 'step': step, 'msg': msg})

        def _prog():
            return _go(2), _render_log(log), gr.update(), gr.update(), gr.update(visible=False), gr.update(), library

        _log('info','Inicio', f"{pending['name']} | {pending['size_mb']:.1f} MB")
        yield _prog()

        try:
            analyzer = MusicAnalyzer(pending['path'])
            analyzer.load_audio()
            _log('ok','Audio', f"SR={analyzer.sr} · {analyzer.y.shape[0]/analyzer.sr:.1f}s")
            yield _prog()

            results = analyzer.analyze()
            _log('ok','Análisis', f"Tono={results['key']['key_en']} · BPM={results['tempo']['bpm']}")
            yield _prog()

            import soundfile as sf_
            _buf = io.BytesIO()
            sf_.write(_buf, analyzer.y, analyzer.sr, format='WAV', subtype='PCM_16')
            audio_bytes = _buf.getvalue()
            _log('ok','WAV', f"{len(audio_bytes)/1_048_576:.2f} MB")
            yield _prog()

            stems = None; model_name = ''
            try:
                from stem_separation import check_demucs
                demucs_ok, _ = check_demucs()
            except Exception:
                demucs_ok = False

            if demucs_ok and sel_stems:
                model_name = 'htdemucs_6s' if any(s in sel_stems for s in ('piano','guitar')) else 'htdemucs_ft'
                _log('info','Demucs', f"{model_name} · {sel_stems}")
                _log('info','GPU', f"Separando en GPU… (~30-60s, no hay actualizaciones intermedias)")
                yield _prog()
                from stem_separation import separate_stems
                all_s = separate_stems(pending['path'], model_name=model_name,
                                       progress_cb=None)  # lambda no es picklable con ZeroGPU
                stems = {k:v for k,v in all_s.items() if k in sel_stems}
                _log('ok','Separación', f"{list(stems.keys())}")
                yield _prog()

            item_id = str(uuid.uuid4())
            item = {
                'id': item_id, 'filename': pending['name'],
                'size_mb': pending['size_mb'], 'ext': pending['ext'],
                'date_added': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'bpm': results.get('tempo',{}).get('bpm',0),
                'key': results.get('key',{}).get('key','—'),
                'key_en': results.get('key',{}).get('key_en','—'),
                'duration': results.get('duration',0),
                'stems_plan': sel_stems, 'model_name': model_name,
                'processing_log': log, 'results': results,
                'stems': stems, 'audio_bytes': audio_bytes,
            }
            _save_item(item)
            library = library + [item]
            _save_library_index(library)
            _log('ok','Completado', f"{round(_time.monotonic()-t0,1)}s total")

            player_html = build_player_html(results, audio_bytes, stems)

            mp3_paths = []
            if stems:
                for key, stem in stems.items():
                    data, mime = _compress_to_mp3(stem['wav_bytes'], '192k')
                    ext2 = 'mp3' if mime == 'audio/mpeg' else 'wav'
                    nm = pending['name'].rsplit('.',1)[0]
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{nm}_{key}.{ext2}')
                    tmp.write(data); tmp.close()
                    mp3_paths.append(tmp.name)

            d = results.get('duration',0)
            proc_time = round(_time.monotonic()-t0,1)
            hdr = (f"<div><b style='font-size:.9rem'>{pending['name']}</b>"
                   f"<span style='color:#888;margin-left:8px;font-size:.75rem'>"
                   f"{results['key']['key_en']} · {results['tempo']['bpm']} BPM · "
                   f"{int(d//60)}:{int(d%60):02d}"
                   f" · ⏱ {proc_time}s</span></div>")

            yield (
                _go(3),
                "",
                gr.update(value=hdr),
                gr.update(value=player_html),
                gr.update(value=mp3_paths or None, visible=bool(mp3_paths)),
                gr.update(value=_kpi_html(results)),
                library,
            )

        except Exception as e:
            _log('error','Error', str(e))
            yield _go(2), _render_log(log)+f"<p style='color:red;margin-top:8px'>{e}</p>", gr.update(), gr.update(), gr.update(visible=False), gr.update(), library

    start_btn.click(
        run_analysis,
        inputs=[pending_st, stem_checks, library_st],
        outputs=[screens, progress_out, detail_header, player_out, download_out, kpi_out, library_st],
    )

    # ── Open library item ─────────────────────────────────────────────────────
    def open_item(item_id, library):
        if not item_id:
            return _go(0), build_library_html(library), library, gr.update(), gr.update(), gr.update(visible=False), gr.update()

        item = next((x for x in library if x['id'] == item_id), None)
        if item and item.get('results') is None:
            item = _load_item_from_disk(item_id)
            library = [item if x['id'] == item_id else x for x in library]

        if item is None:
            return _go(0), build_library_html(library), library, gr.update(), gr.update(), gr.update(visible=False), gr.update()

        results = item.get('results')
        stems   = item.get('stems')
        ab      = item.get('audio_bytes')

        player_html = build_player_html(results, ab, stems) if (results and ab) else ""

        mp3_paths = []
        if stems:
            for key, stem in stems.items():
                data, mime = _compress_to_mp3(stem['wav_bytes'], '192k')
                ext2 = 'mp3' if mime == 'audio/mpeg' else 'wav'
                nm = item['filename'].rsplit('.',1)[0]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{nm}_{key}.{ext2}')
                tmp.write(data); tmp.close()
                mp3_paths.append(tmp.name)

        d = item.get('duration', 0)
        hdr = (f"<div><b style='font-size:.9rem'>{item['filename']}</b>"
               f"<span style='color:#888;margin-left:8px;font-size:.75rem'>"
               f"{item.get('key_en','—')} · {item.get('bpm','—')} BPM · "
               f"{int(d//60)}:{int(d%60):02d}</span></div>")

        return (
            _go(3),
            build_library_html(library),
            library,
            gr.update(value=hdr),
            gr.update(value=player_html),
            gr.update(value=mp3_paths or None, visible=bool(mp3_paths)),
            gr.update(value=_kpi_html(results) if results else ""),
        )

    item_id_box.change(
        open_item,
        inputs=[item_id_box, library_st],
        outputs=[screens, lib_html, library_st, detail_header, player_out, download_out, kpi_out],
    )


if __name__ == "__main__":
    import os
    if os.environ.get('SPACE_ID'):
        demo.launch(ssr_mode=False, css=GRADIO_CSS)
    else:
        demo.launch(server_port=7861, ssr_mode=False, css=GRADIO_CSS)
