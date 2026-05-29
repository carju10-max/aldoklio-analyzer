"""
stem_separation.py — Separación de pistas con Demucs
Separa el audio en: Voz, Batería, Bajo y Otro.
"""

import io
import warnings
import numpy as np
import soundfile as sf

warnings.filterwarnings('ignore')

# En HuggingFace Spaces con ZeroGPU, @spaces.GPU asigna GPU para la función.
# Localmente (sin el paquete spaces), es un no-op transparente.
try:
    import spaces
except ImportError:
    import types as _types
    spaces = _types.SimpleNamespace(GPU=lambda **kw: (lambda f: f))

# Caché de modelos en memoria — evita recargar el modelo entre análisis
_MODEL_CACHE: dict = {}


def _preload_default_model():
    """Precarga htdemucs_ft en background al importar el módulo."""
    try:
        import torch
        from demucs.pretrained import get_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        key = f'htdemucs_ft_{device}'
        if key not in _MODEL_CACHE:
            m = get_model('htdemucs_ft')
            m.eval()
            m.to(device)
            _MODEL_CACHE[key] = m
    except Exception:
        pass  # silencioso — si falla, separate_stems lo carga normalmente


import threading as _threading
_threading.Thread(target=_preload_default_model, daemon=True).start()

# ── Metadatos de cada pista ───────────────────────────────────────────────────
STEMS_INFO = {
    'vocals': {'name_es': 'Voz',      'icon': 'V', 'color': '#c8d4e0'},
    'drums':  {'name_es': 'Batería',  'icon': 'D', 'color': '#8090a0'},
    'bass':   {'name_es': 'Bajo',     'icon': 'B', 'color': '#687888'},
    'other':  {'name_es': 'Otro',     'icon': 'O', 'color': '#a0b0c0'},
    'guitar': {'name_es': 'Guitarra', 'icon': 'G', 'color': '#90a0b0'},
    'piano':  {'name_es': 'Piano',    'icon': 'P', 'color': '#b0c0d0'},
}

# Modelos disponibles (clave → nombre demucs)
# htdemucs_ft = fine-tuned, más rápido y preciso que htdemucs base
MODELS = {
    '4 pistas  —  Voz / Batería / Bajo / Otro':      'htdemucs_ft',
    '6 pistas  —  + Guitarra / Piano (más lento)':   'htdemucs_6s',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_demucs() -> tuple[bool, str]:
    """
    Devuelve (True, '') si demucs y torch están disponibles,
    o (False, mensaje) si no.
    """
    try:
        import torch    # noqa: F401
        import demucs   # noqa: F401
        return True, ""
    except ImportError as exc:
        missing = "demucs" if "demucs" in str(exc) else "torch"
        return False, (
            f"Falta el paquete `{missing}`.  \n"
            "**Instálalo con:**\n\n"
            "```\npip install demucs\n```\n\n"
            "_(Puede tardar unos minutos; incluye PyTorch automáticamente)_"
        )


def _load_stereo(path: str, target_sr: int) -> np.ndarray:
    """
    Carga audio como array numpy estéreo (2, samples),
    remuestreado a target_sr.
    """
    import librosa

    try:
        data, sr = sf.read(path, always_2d=True)
        wav = data.T.astype(np.float32)   # (ch, samples)
    except Exception:
        wav, sr = librosa.load(path, sr=None, mono=False)
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]

    # Resamplear si hace falta
    if int(sr) != int(target_sr):
        wav = np.stack([
            librosa.resample(ch, orig_sr=int(sr), target_sr=int(target_sr))
            for ch in wav
        ])

    # Garantizar estéreo 2 canales
    if wav.shape[0] == 1:
        wav = np.concatenate([wav, wav], axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    return wav.astype(np.float32)


# ── Función principal ─────────────────────────────────────────────────────────

@spaces.GPU(duration=60)
def separate_stems(
    file_path: str,
    model_name: str = 'htdemucs',
    progress_cb=None,
) -> dict:
    """
    Separa el audio en pistas individuales usando Demucs.

    Parámetros
    ----------
    file_path   : ruta al archivo de audio/video
    model_name  : nombre del modelo ('htdemucs' | 'htdemucs_6s')
    progress_cb : función opcional (pct: int, msg: str) → None

    Devuelve
    --------
    dict  {stem_key: {name_es, icon, color, audio_mono, sr, wav_bytes}}
    """

    def _cb(pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    # ── Importar dependencias ──────────────────────────────────────────────────
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as exc:
        raise ImportError(
            "demucs no disponible. Instálalo con:\n  pip install demucs"
        ) from exc

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_key = f"{model_name}_{device}"

    if cache_key in _MODEL_CACHE:
        _cb(5, "⚡  Modelo en caché (carga instantánea)…")
        model = _MODEL_CACHE[cache_key]
    else:
        _cb(5, f"⬇️  Cargando modelo '{model_name}' (primera vez descarga ~300 MB)…")
        model = get_model(model_name)
        model.eval()
        model.to(device)
        _MODEL_CACHE[cache_key] = model

    _cb(18, "📂  Cargando y preparando el audio…")
    wav = _load_stereo(file_path, model.samplerate)
    duration_s = wav.shape[1] / model.samplerate

    # Normalizar para estabilidad numérica
    ref_mean = float(wav.mean())
    ref_std  = float(wav.std()) + 1e-8
    wav_norm = (wav - ref_mean) / ref_std

    sr         = model.samplerate
    n_samples  = wav_norm.shape[1]
    duration_s = n_samples / sr

    _cb(30, f"🤖  Separando con IA en CPU — {duration_s:.0f}s de audio"
        f"  (~{max(1, int(duration_s/60)*3)}–{max(2, int(duration_s/60)*6)} min)…")

    # Para limitar pico de RAM dividimos el audio en chunks a nivel numpy.
    # Cada chunk se procesa independientemente y se ensambla con crossfade.
    # chunk_sec=90s → ~400MB pico RAM por chunk (funciona para htdemucs_6s)
    chunk_sec  = 90
    overlap_s  = 3
    chunk_samp = int(chunk_sec * sr)
    step_samp  = int((chunk_sec - overlap_s) * sr)
    starts     = list(range(0, n_samples, step_samp))
    n_chunks   = len(starts)
    n_stems    = len(model.sources)

    out_full   = np.zeros((n_stems, 2, n_samples), dtype=np.float32)
    wt_full    = np.zeros((n_stems, 2, n_samples), dtype=np.float32)
    fade       = int(overlap_s * sr)

    with torch.no_grad():
        for ci, start in enumerate(starts):
            end   = min(start + chunk_samp, n_samples)
            chunk = wav_norm[:, start:end]
            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)

            out = apply_model(
                model, tensor,
                device=device,
                overlap=0.1,
                shifts=0,
                num_workers=0,
            )[0].cpu().numpy()  # (n_stems, 2, T_chunk)

            clen = end - start
            w    = np.ones(clen, dtype=np.float32)
            if ci > 0 and fade < clen:
                w[:fade] = np.linspace(0, 1, fade)
            if ci < n_chunks - 1 and fade < clen:
                w[-fade:] = np.linspace(1, 0, fade)

            out_full[:, :, start:end] += out[:, :, :clen] * w
            wt_full[:, :, start:end]  += w

            pct = 30 + int((ci + 1) / n_chunks * 56)
            _cb(pct, f"⚙️  Chunk {ci+1}/{n_chunks}"
                f" ({int(start/sr)}s→{int(end/sr)}s)  RAM controlada…")

    wt_full = np.clip(wt_full, 1e-8, None)
    # Desnormalizar directamente en numpy
    sources = (out_full / wt_full) * ref_std + ref_mean  # (n_stems, 2, T)

    _cb(88, "🎚️  Convirtiendo y empaquetando pistas…")

    results = {}
    for i, stem_key in enumerate(model.sources):
        info = STEMS_INFO.get(stem_key, {
            'name_es': stem_key.capitalize(),
            'icon': '🎵',
            'color': '#9e9e9e',
        })

        audio_np  = sources[i]   # (2, T) estéreo — ya es numpy
        audio_mono = audio_np.mean(axis=0)      # (T,)  mono para visualizar

        # Serializar a WAV en memoria
        buf = io.BytesIO()
        sf.write(buf, audio_np.T, model.samplerate,
                 format='WAV', subtype='PCM_16')

        results[stem_key] = {
            'name_es':    info['name_es'],
            'icon':       info['icon'],
            'color':      info['color'],
            'audio_mono': audio_mono,
            'sr':         model.samplerate,
            'wav_bytes':  buf.getvalue(),
        }

    _cb(100, "✅  ¡Separación completada!")
    return results
