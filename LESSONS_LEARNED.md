# Lecciones aprendidas — Deploy en HuggingFace Spaces

Historial de errores encontrados al desplegar este proyecto en HF Spaces con ZeroGPU y Gradio 6.

---

## 1. `@spaces.GPU` debe estar literal en `app.py`

**Error:** `RUNTIME_ERROR: No @spaces.GPU function detected during startup`

**Causa:** HF hace un scan estático de `app.py` al arrancar buscando el string literal `@spaces.GPU`. El patrón de alias no funciona aunque sea funcionalmente equivalente.

```python
# ❌ No detectado por HF
_gpu = spaces.GPU(duration=300)

@_gpu
def separate_stems(...): ...
```

```python
# ✅ Detectado correctamente
try:
    import spaces
    @spaces.GPU(duration=1)
    def _gpu_warmup(): pass
except ImportError:
    pass
```

**Commits:** `10a8c10`, `c07a5bc`, `40b435e`

---

## 2. `ssr_mode=False` obligatorio en HF Spaces con Gradio 6

**Error:** Space se cargaba pero crasheaba inmediatamente con un error del proxy Node.

**Causa:** Gradio 6 activa SSR (Server-Side Rendering) por defecto. El proxy de HF Spaces no soporta el modo SSR de Gradio y crashea.

```python
# ✅ Correcto para HF
if os.environ.get('SPACE_ID'):
    demo.launch(ssr_mode=False)
else:
    demo.launch(server_port=7861, ssr_mode=False)
```

**Commits:** `4804617`, `dce9b48`

---

## 3. El parámetro `css` va en `gr.Blocks()`, no en `launch()`

**Error:** `TypeError: launch() got an unexpected keyword argument 'css'`

**Causa:** En Gradio 6, `css` se configura en la construcción del bloque, no al lanzar.

```python
# ❌ Error en Gradio 6
demo.launch(css="...")

# ✅ Correcto
with gr.Blocks(css="...") as demo:
    ...
```

**Commit:** `b755705`

---

## 4. Puerto 7860 en HF, custom solo en local

**Error:** Space no respondía en el puerto esperado.

**Causa:** HF Spaces espera la app en el puerto 7860 por defecto. Fijar `server_port=7861` rompía el routing.

```python
if os.environ.get('SPACE_ID'):
    demo.launch(ssr_mode=False)          # HF usa 7860 por defecto
else:
    demo.launch(server_port=7861, ...)   # local: puerto custom
```

**Commit:** `0110f77`

---

## 5. `requirements.txt` — solo dependencias usadas

**Error:** Build lento o con conflictos silenciosos.

**Causa:** El proyecto tenía `streamlit` y `plotly` en `requirements.txt` como herencia de la versión anterior. Ninguno se usa en la versión Gradio. Streamlit en particular entra en conflicto con el entorno de Gradio.

**Commit:** `5750be8`

---

## 6. Verificar qué SHA está desplegado realmente

Cuando un build falla o parece no actualizarse, HF puede seguir corriendo un SHA anterior.
Para saber cuál está activo:

```bash
curl -s https://huggingface.co/api/spaces/CarlosDiazSalazar/aldoklio-analyzer | python -m json.tool | findstr sha
```

Si el SHA no coincide con el último commit local, HF todavía está procesando. Esperar 2–5 min y reintentar.

---

## Resumen del proceso de deploy

```bash
# Workflow estándar
git push origin main   # GitHub (backup)
git push hf main       # HF Spaces (triggeriza rebuild automático ~3-8 min)

# Verificar estado
curl -s https://huggingface.co/api/spaces/CarlosDiazSalazar/aldoklio-analyzer
# Buscar: "stage": "RUNNING"
```

**URLs del proyecto:**
- Space: https://carlosdiazsalazar-aldoklio-analyzer.hf.space
- Repo HF: https://huggingface.co/spaces/CarlosDiazSalazar/aldoklio-analyzer
- GitHub: https://github.com/carju10-max/aldoklio-analyzer
