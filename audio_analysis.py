"""
audio_analysis.py - Motor de análisis musical
Usa librosa para extraer información musical de archivos de audio.
"""

import os
import librosa
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ── Constantes ───────────────────────────────────────────────────────────────

NOTE_NAMES_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_ES = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']

# Perfiles de Krumhansl-Schmuckler para detección de tonalidad
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Plantillas de acordes (intervalos en semítonos desde la raíz)
CHORD_TEMPLATES = {
    'maj':  [0, 4, 7],           # mayor
    'min':  [0, 3, 7],           # menor
    '7':    [0, 4, 7, 10],       # dominante 7ª
    'maj7': [0, 4, 7, 11],       # mayor 7ª
    'm7':   [0, 3, 7, 10],       # menor 7ª
    'dim':  [0, 3, 6],           # disminuido
    'aug':  [0, 4, 8],           # aumentado
    'sus4': [0, 5, 7],           # suspendida 4ª
    'sus2': [0, 2, 7],           # suspendida 2ª
}

CHORD_NAME_SUFFIX_ES = {
    'maj':  '',
    'min':  'm',
    '7':    '7',
    'maj7': 'M7',
    'm7':   'm7',
    'dim':  'dim',
    'aug':  'aug',
    'sus4': 'sus4',
    'sus2': 'sus2',
}

SEGMENT_COLORS = {
    'Intro':   '#ff7043',
    'Verso':   '#42a5f5',
    'Coro':    '#66bb6a',
    'Puente':  '#ffa726',
    'Solo':    '#ec407a',
    'Outro':   '#ab47bc',
    'Cuerpo':  '#26c6da',
}


# ── Clase principal ───────────────────────────────────────────────────────────

class MusicAnalyzer:
    """Analiza un archivo de audio y extrae información musical completa."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.y: np.ndarray | None = None
        self.sr: int | None = None
        self.duration: float | None = None

    # ── Carga ──────────────────────────────────────────────────────────────

    # ── Extracción de audio desde video ────────────────────────────────────

    @staticmethod
    def is_video_file(path: str) -> bool:
        """Devuelve True si la extensión corresponde a un formato de video."""
        VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv',
                     '.flv', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'}
        ext = os.path.splitext(path)[1].lower()
        return ext in VIDEO_EXT

    @staticmethod
    def _find_ffmpeg() -> str | None:
        """
        Devuelve la ruta completa a ffmpeg.exe buscando en:
          1. PATH del sistema (shutil.which)
          2. Carpeta de paquetes de winget (instalación Gyan.FFmpeg)
        """
        import shutil
        found = shutil.which("ffmpeg")
        if found:
            return found
        winget_base = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
        if os.path.isdir(winget_base):
            for root, _dirs, files in os.walk(winget_base):
                for f in files:
                    if f.lower() == "ffmpeg.exe":
                        return os.path.join(root, f)
        return None

    @staticmethod
    def extract_audio_from_video(video_path: str, out_wav: str) -> str:
        """Extrae la pista de audio de un video a WAV usando ffmpeg."""
        import subprocess
        ffmpeg_bin = MusicAnalyzer._find_ffmpeg()   # ← usa _find_ffmpeg, no shutil.which
        if not ffmpeg_bin:
            raise EnvironmentError(
                "ffmpeg no encontrado. Instálalo con:\n"
                "  winget install --id Gyan.FFmpeg -e\n"
                "y luego reinicia el programa."
            )
        cmd = [ffmpeg_bin, "-y", "-i", video_path,
               "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", out_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg falló.\nDetalle: {result.stderr[-600:]}")
        return out_wav

    @staticmethod
    def _pydub_to_numpy(audio_seg) -> np.ndarray:
        """Convierte un AudioSegment de pydub a array numpy normalizado [-1, 1]."""
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (audio_seg.sample_width * 8 - 1))
        return samples / max_val

    def load_audio(self) -> tuple[np.ndarray, int]:
        """
        Carga audio o extrae audio de video.
        Orden de intentos:
          1. pydub + ffmpeg  (soporta MP3, M4A, video, etc.)
          2. soundfile       (WAV, FLAC, OGG sin ffmpeg)
          3. librosa nativo  (fallback general)
        """
        import os as _os
        import tempfile

        TARGET_SR = 22050
        tmp_wav   = None   # archivo temporal creado por nosotros (si aplica)

        # ── 0. Apuntar ffmpeg a pydub ──────────────────────────────────────
        ffmpeg_bin = self._find_ffmpeg()
        if ffmpeg_bin:
            import pydub
            pydub.AudioSegment.converter = ffmpeg_bin
            pydub.AudioSegment.ffmpeg    = ffmpeg_bin
            pydub.AudioSegment.ffprobe   = ffmpeg_bin.replace("ffmpeg", "ffprobe")

        actual_path = self.file_path

        try:
            # ── 1. pydub (MP3, M4A, video, y todo lo que ffmpeg soporte) ──
            if ffmpeg_bin or not self.is_video_file(self.file_path):
                from pydub import AudioSegment
                try:
                    audio_seg = AudioSegment.from_file(actual_path)
                    audio_seg = audio_seg.set_frame_rate(TARGET_SR).set_channels(1)
                    self.y  = self._pydub_to_numpy(audio_seg)
                    self.sr = TARGET_SR
                    # Éxito por pydub
                    self.duration = librosa.get_duration(y=self.y, sr=self.sr)
                    if self.duration < 2.0:
                        raise ValueError("El archivo es demasiado corto (mínimo 2 s).")
                    return self.y, self.sr
                except Exception:
                    pass  # intentar siguiente método

            # ── 2. Si es video y pydub falló, usar ffmpeg directamente ────
            if self.is_video_file(self.file_path):
                if not ffmpeg_bin:
                    raise EnvironmentError(
                        "ffmpeg no encontrado. Instálalo con:\n"
                        "  winget install --id Gyan.FFmpeg -e"
                    )
                fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
                _os.close(fd)
                self.extract_audio_from_video(actual_path, tmp_wav)
                actual_path = tmp_wav

            # ── 3. soundfile (WAV, FLAC, OGG) ────────────────────────────
            try:
                import soundfile as sf
                data, sr = sf.read(actual_path, always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)
                if sr != TARGET_SR:
                    data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
                self.y, self.sr = data, TARGET_SR
            except Exception:
                # ── 4. librosa nativo (último recurso) ───────────────────
                self.y, self.sr = librosa.load(actual_path, sr=TARGET_SR, mono=True)

        finally:
            if tmp_wav and _os.path.exists(tmp_wav):
                try:
                    _os.unlink(tmp_wav)
                except OSError:
                    pass

        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        if self.duration < 2.0:
            raise ValueError("El archivo de audio es demasiado corto (mínimo 2 segundos).")
        return self.y, self.sr

    # ── Tonalidad ──────────────────────────────────────────────────────────

    def detect_key(self) -> dict:
        """Detecta la tonalidad usando el algoritmo de Krumhansl-Schmuckler."""
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, bins_per_octave=36)
        chroma_mean = np.mean(chroma, axis=1)

        # Normalizar
        chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-10)

        best_key, best_mode, best_corr = 0, 'major', -np.inf

        for i in range(12):
            maj_rot = np.roll(MAJOR_PROFILE, i)
            min_rot = np.roll(MINOR_PROFILE, i)

            maj_norm = maj_rot / maj_rot.sum()
            min_norm = min_rot / min_rot.sum()

            try:
                c_maj, _ = pearsonr(chroma_norm, maj_norm)
                c_min, _ = pearsonr(chroma_norm, min_norm)
            except Exception:
                c_maj = c_min = 0.0

            if c_maj > best_corr:
                best_corr, best_key, best_mode = c_maj, i, 'major'
            if c_min > best_corr:
                best_corr, best_key, best_mode = c_min, i, 'minor'

        note_es = NOTE_NAMES_ES[best_key]
        note_en = NOTE_NAMES_EN[best_key]
        mode_es = 'Mayor' if best_mode == 'major' else 'menor'
        mode_en = 'Major' if best_mode == 'major' else 'Minor'

        return {
            'key':        f"{note_es} {mode_es}",
            'key_en':     f"{note_en} {mode_en}",
            'root':        best_key,
            'root_name':   note_en,
            'root_name_es': note_es,
            'mode':        best_mode,
            'confidence':  round(max(0.0, float(best_corr)) * 100, 1),
            'chroma_mean': chroma_mean.tolist(),
        }

    # ── Tempo ──────────────────────────────────────────────────────────────

    def detect_tempo(self) -> dict:
        """Detecta el BPM y las posiciones de cada pulso."""
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=512)
        tempo_arr, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=self.sr, hop_length=512, start_bpm=90
        )
        tempo = float(np.atleast_1d(tempo_arr)[0])

        # Normalizar al rango 60–120 BPM (pulso humano percibido).
        # start_bpm=90 ya sesga hacia tempos reales, pero por si acaso:
        while tempo > 120:
            tempo /= 2
            beats = beats[::2]
        while tempo < 60:
            tempo *= 2

        beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=512)
        beat_offset = float(beat_times[0]) if len(beat_times) > 0 else 0.0

        return {
            'bpm':         round(tempo, 1),
            'beat_times':  beat_times.tolist(),
            'beats':       beats.tolist(),
            'beat_offset': round(beat_offset, 3),
        }

    # ── Compás ─────────────────────────────────────────────────────────────

    def detect_time_signature(self, beat_times: list) -> dict:
        """Estima el compás analizando el patrón de energía en los pulsos."""
        TIME_SIG_INFO = {
            2: ('2/4', 2, 'Compás de 2/4 — Marcha / Polka'),
            3: ('3/4', 3, 'Compás de 3/4 — Vals'),
            4: ('4/4', 4, 'Compás de 4/4 — Rock / Pop (más común)'),
            6: ('6/8', 6, 'Compás de 6/8 — Compuesto / Balada'),
        }

        if len(beat_times) < 8:
            ts, bpm, desc = TIME_SIG_INFO[4]
            return {'time_sig': ts, 'beats_per_measure': bpm, 'description': desc}

        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop_length)

        bt = np.array(beat_times)
        bf = librosa.time_to_frames(bt, sr=self.sr, hop_length=hop_length)
        bf = np.clip(bf, 0, len(onset_env) - 1)
        beat_strength = onset_env[bf]

        scores: dict[int, float] = {}
        for meter in [2, 3, 4, 6]:
            n_complete = len(beat_strength) // meter
            if n_complete < 4:
                continue
            mat = beat_strength[:n_complete * meter].reshape(n_complete, meter)
            profile = mat.mean(axis=0)
            profile /= (profile.sum() + 1e-10)
            # Cuánto mayor es el tiempo 1 respecto al resto
            scores[meter] = float(profile[0] - profile[1:].mean())

        best_meter = max(scores, key=scores.get) if scores else 4
        ts, bpm, desc = TIME_SIG_INFO[best_meter]
        return {'time_sig': ts, 'beats_per_measure': bpm, 'description': desc}

    # ── Escala ─────────────────────────────────────────────────────────────

    def detect_scale(self, key_info: dict) -> dict:
        """Determina la escala a partir de la tonalidad detectada."""
        root = key_info['root']
        mode = key_info['mode']

        if mode == 'major':
            intervals = [0, 2, 4, 5, 7, 9, 11]
            mode_name = 'Mayor (Jónico)'
            formula = 'T-T-S-T-T-T-S'
        else:
            intervals = [0, 2, 3, 5, 7, 8, 10]
            mode_name = 'Menor natural (Eólico)'
            formula = 'T-S-T-T-S-T-T'

        notes_idx = [(root + i) % 12 for i in intervals]
        notes_es = [NOTE_NAMES_ES[n] for n in notes_idx]
        notes_en = [NOTE_NAMES_EN[n] for n in notes_idx]

        scale_name_es = f"Escala {key_info['key']}"

        return {
            'scale_name':  scale_name_es,
            'mode_name':   mode_name,
            'formula':     formula,
            'notes':       notes_es,
            'notes_en':    notes_en,
            'description': f"{scale_name_es}: {' — '.join(notes_es)}",
        }

    # ── Acordes ────────────────────────────────────────────────────────────

    def detect_chords(self) -> dict:
        """
        Detecta acordes con alta precisión usando 5 mejoras clave:
          1. HPSS — aislar componente armónica (elimina batería/ruido)
          2. chroma_cens — específico para reconocimiento de acordes
          3. Grupos de 2 beats — segmentos más estables y menos ruidosos
          4. Umbral de confianza — rechazar acordes inciertos
          5. Duración mínima — filtrar cambios demasiado rápidos (<1 seg)
        """
        from collections import Counter

        HOP = 2048   # ~93 ms/frame — mucho más suave que el 512 anterior

        # ── 1. Separar componente armónica (quita batería, ruido, percusión) ──
        y_harmonic, _ = librosa.effects.hpss(self.y, margin=6.0)

        # ── 2. Chroma CENS — diseñado para reconocimiento de acordes ──────────
        #    (más robusto al timbre que chroma_cqt)
        chroma = librosa.feature.chroma_cens(
            y=y_harmonic, sr=self.sr,
            hop_length=HOP,
            win_len_smooth=21,   # suavizado moderado — menos blur que 41
        )

        # ── 3. Construir plantillas de acordes ────────────────────────────────
        all_templates: dict[str, np.ndarray] = {}
        for root in range(12):
            for chord_type, intervals in CHORD_TEMPLATES.items():
                suffix = CHORD_NAME_SUFFIX_ES[chord_type]
                name   = NOTE_NAMES_ES[root] + suffix
                vec    = np.zeros(12)
                for iv in intervals:
                    vec[(root + iv) % 12] = 1.0
                all_templates[name] = vec / (np.linalg.norm(vec) + 1e-10)

        # ── 4. Beat tracking ──────────────────────────────────────────────────
        _, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=HOP)

        if len(beats) >= 4:
            # Beat a beat — detecta cambios de acorde cada beat (no cada 2)
            beat_groups = beats
            times_sync  = librosa.frames_to_time(beat_groups, sr=self.sr, hop_length=HOP)
            chroma_sync = librosa.util.sync(chroma, beat_groups, aggregate=np.median)
        else:
            # Fallback: ventanas de 1 segundo
            hop_1s = max(1, int(self.sr / HOP))
            chroma_sync = librosa.util.sync(
                chroma,
                np.arange(0, chroma.shape[1], hop_1s),
                aggregate=np.median,
            )
            n = chroma_sync.shape[1]
            times_sync = np.linspace(0, self.duration, n)

        # ── 5. Acorde más probable, con umbral de confianza ───────────────────
        MIN_SCORE = 0.65   # descartar si el match es débil
        raw_chords: list[str] = []

        for i in range(chroma_sync.shape[1]):
            frame = chroma_sync[:, i]
            if frame.sum() < 0.08:
                raw_chords.append('N/C')
                continue
            frame_n = frame / (np.linalg.norm(frame) + 1e-10)
            best, best_score = 'N/C', -1.0
            for name, tmpl in all_templates.items():
                score = float(np.dot(frame_n, tmpl))
                if score > best_score:
                    best_score, best = score, name
            raw_chords.append(best if best_score >= MIN_SCORE else 'N/C')

        # ── 6. Suavizado moderado — ventana de ±1 ────────────────────────────
        W = 1
        smoothed: list[str] = []
        for i in range(len(raw_chords)):
            window     = raw_chords[max(0, i - W): i + W + 1]
            candidates = [c for c in window if c != 'N/C']
            smoothed.append(
                Counter(candidates).most_common(1)[0][0] if candidates else 'N/C'
            )

        # ── 7. Extraer cambios + filtrar duración mínima (1 segundo) ─────────
        MIN_DUR = 0.3   # acordes que duran menos de 0.3 s = artefacto, se descarta

        raw_changes: list[dict] = []
        prev = None
        for i, chord in enumerate(smoothed):
            if chord != prev and i < len(times_sync):
                t = float(times_sync[i])
                raw_changes.append({
                    'time':     round(t, 2),
                    'chord':    chord,
                    'time_fmt': self._fmt_time(t),
                })
                prev = chord

        # Aplicar filtro de duración mínima
        chord_changes: list[dict] = []
        for i, ch in enumerate(raw_changes):
            end_t = raw_changes[i + 1]['time'] if i + 1 < len(raw_changes) else self.duration
            if (end_t - ch['time']) >= MIN_DUR:
                chord_changes.append(ch)

        # Si el filtro dejó muy pocos, usamos sin filtro de duración
        if len(chord_changes) < 3:
            chord_changes = raw_changes

        counts = Counter(c for c in smoothed if c != 'N/C')

        return {
            'chord_sequence': smoothed,
            'chord_changes':  chord_changes,
            'top_chords':     counts.most_common(12),
        }

    # ── Estructura ─────────────────────────────────────────────────────────

    def detect_structure(self) -> dict:
        """Detecta la estructura de la canción (intro, verso, coro, etc.)."""
        hop_length = 512

        # Combinar MFCC + Cromas como descriptor
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20, hop_length=hop_length)
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, hop_length=hop_length)

        features = np.vstack([
            librosa.util.normalize(mfcc, axis=1),
            librosa.util.normalize(chroma, axis=1),
        ])

        # Número de segmentos objetivo
        k = min(8, max(3, int(self.duration / 25)))

        try:
            boundaries = librosa.segment.agglomerative(features, k)
        except Exception:
            n_frames = features.shape[1]
            boundaries = np.linspace(0, n_frames, k + 1, dtype=int)

        boundary_times = librosa.frames_to_time(boundaries, sr=self.sr, hop_length=hop_length)
        boundary_times = np.clip(boundary_times, 0, self.duration)

        # Calcular energía RMS de cada segmento
        energies: list[float] = []
        for i in range(len(boundary_times) - 1):
            s = int(boundary_times[i] * self.sr)
            e = int(boundary_times[i + 1] * self.sr)
            seg = self.y[s:e]
            if len(seg) > 0:
                energies.append(float(np.mean(librosa.feature.rms(y=seg))))
            else:
                energies.append(0.0)

        max_e = max(energies) if energies else 1.0
        norm_energies = [e / (max_e + 1e-10) for e in energies]

        labels = self._label_segments(norm_energies)

        segments: list[dict] = []
        for i, (label, nrg) in enumerate(zip(labels, norm_energies)):
            start = float(boundary_times[i])
            end   = float(boundary_times[i + 1]) if i + 1 < len(boundary_times) else self.duration
            dur   = end - start
            if dur < 1.0:
                continue
            segments.append({
                'label':     label,
                'start':     round(start, 2),
                'end':       round(end, 2),
                'duration':  round(dur, 2),
                'energy':    round(nrg, 3),
                'start_fmt': self._fmt_time(start),
                'end_fmt':   self._fmt_time(end),
                'color':     SEGMENT_COLORS.get(label, '#9e9e9e'),
            })

        return {
            'segments':        segments,
            'n_segments':      len(segments),
            'boundary_times':  boundary_times.tolist(),
        }

    def _label_segments(self, energies: list[float]) -> list[str]:
        """Asigna etiquetas heurísticas a los segmentos."""
        n = len(energies)
        if n == 0:
            return []
        if n == 1:
            return ['Cuerpo']

        labels = ['Verso'] * n
        labels[0]  = 'Intro'
        labels[-1] = 'Outro'

        if n <= 2:
            return labels

        arr = np.array(energies)
        hi = float(np.percentile(arr, 72))
        lo = float(np.percentile(arr, 28))

        chorus_count = 0
        for i in range(1, n - 1):
            if energies[i] >= hi:
                labels[i] = 'Coro'
                chorus_count += 1
            elif energies[i] <= lo and n > 4:
                labels[i] = 'Puente'

        # Si no hay coro detectado y la canción es larga, usar el segmento central
        if chorus_count == 0 and n >= 4:
            mid = n // 2
            if labels[mid] not in ('Intro', 'Outro'):
                labels[mid] = 'Coro'

        return labels

    # ── Datos de visualización ─────────────────────────────────────────────

    def get_waveform_data(self, n_points: int = 1200) -> dict:
        """Devuelve la forma de onda reducida para visualización."""
        step = max(1, len(self.y) // n_points)
        y_vis = self.y[::step]
        t_vis = np.linspace(0, self.duration, len(y_vis))
        return {
            'times':      t_vis.tolist(),
            'amplitudes': y_vis.tolist(),
            'duration':   self.duration,
        }

    def get_spectrogram_data(self) -> dict:
        """Devuelve datos del espectrograma Mel."""
        hop = 512
        mel = librosa.feature.melspectrogram(
            y=self.y, sr=self.sr, n_fft=2048, hop_length=hop, n_mels=80
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Reducir resolución temporal para el navegador
        max_t = 300
        if mel_db.shape[1] > max_t:
            step = mel_db.shape[1] // max_t
            mel_db = mel_db[:, ::step]

        times = np.linspace(0, self.duration, mel_db.shape[1]).tolist()
        freqs = librosa.mel_frequencies(n_mels=80, fmin=0, fmax=self.sr / 2).tolist()

        return {
            'spectrogram': mel_db.tolist(),
            'times':       times,
            'freqs':       freqs,
        }

    def get_chromagram_data(self) -> dict:
        """Devuelve el cromograma."""
        hop = 512
        chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr, hop_length=hop)

        max_t = 300
        if chroma.shape[1] > max_t:
            step = chroma.shape[1] // max_t
            chroma = chroma[:, ::step]

        times = np.linspace(0, self.duration, chroma.shape[1]).tolist()
        return {
            'chroma': chroma.tolist(),
            'times':  times,
            'notes':  NOTE_NAMES_EN,
        }

    # ── Análisis completo ──────────────────────────────────────────────────

    def analyze(self, progress_cb=None) -> dict:
        """Ejecuta el análisis completo y devuelve todos los resultados."""

        def _cb(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        _cb(5,  "Cargando audio…")
        self.load_audio()

        _cb(20, "Detectando tonalidad…")
        key = self.detect_key()

        _cb(35, "Detectando tempo…")
        tempo = self.detect_tempo()

        _cb(45, "Analizando compás…")
        ts = self.detect_time_signature(tempo['beat_times'])

        _cb(50, "Identificando escala…")
        scale = self.detect_scale(key)

        _cb(60, "Detectando acordes…")
        chords = self.detect_chords()

        _cb(75, "Analizando estructura…")
        structure = self.detect_structure()

        _cb(85, "Preparando visualizaciones…")
        waveform    = self.get_waveform_data()
        spectrogram = self.get_spectrogram_data()
        chromagram  = self.get_chromagram_data()

        _cb(100, "¡Análisis completado!")

        return {
            'key':           key,
            'tempo':         tempo,
            'time_signature': ts,
            'scale':         scale,
            'chords':        chords,
            'structure':     structure,
            'waveform':      waveform,
            'spectrogram':   spectrogram,
            'chromagram':    chromagram,
            'duration':      self.duration,
            'sample_rate':   self.sr,
            'file_name':     self.file_path,
        }

    # ── Utilidades ─────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
