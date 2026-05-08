# WallForge Studio

Editor 2D de entornos de paredes para simulación robótica con MuJoCo.
Diseñado para el flujo de trabajo del repositorio **g1man** (robot humanoide Unitree G1).

---

## Tabla de contenidos

1. [¿Qué es WallForge Studio?](#1-qué-es-wallforge-studio)
2. [Instalación y dependencias](#2-instalación-y-dependencias)
3. [Cómo ejecutar](#3-cómo-ejecutar)
4. [Interfaz de usuario](#4-interfaz-de-usuario)
5. [Herramientas de edición](#5-herramientas-de-edición)
6. [Sistema de snap](#6-sistema-de-snap)
7. [Plano de fondo (floor plan)](#7-plano-de-fondo-floor-plan)
8. [Detección automática de paredes](#8-detección-automática-de-paredes)
9. [Exportación a MuJoCo XML](#9-exportación-a-mujoco-xml)
10. [Gestión de proyectos](#10-gestión-de-proyectos)
11. [Atajos de teclado](#11-atajos-de-teclado)
12. [Arquitectura del código](#12-arquitectura-del-código)
13. [Sistema de coordenadas](#13-sistema-de-coordenadas)
14. [Formato de archivo `.wfp`](#14-formato-de-archivo-wfp)

---

## 1. ¿Qué es WallForge Studio?

WallForge Studio es una herramienta gráfica de escritorio que permite:

- **Dibujar paredes** manualmente sobre un canvas 2D con snap a rejilla y a extremos
- **Importar un plano arquitectónico** (imagen PNG/JPG/PDF) como capa de fondo
- **Calibrar la escala** del plano visualmente, dibujando líneas de medida sobre la imagen
- **Detectar paredes automáticamente** a partir de la imagen mediante un pipeline de visión por computador (Hough + componentes conectadas)
- **Editar paredes con precisión** arrastrando sus extremos individualmente
- **Exportar la escena a MuJoCo XML** en dos modos: escena básica genérica o directamente compatible con el repositorio g1man

El formato de proyecto nativo es `.wfp` (WallForge Project, JSON).

---

## 2. Instalación y dependencias

### Dependencias obligatorias

```bash
pip install pillow opencv-python numpy
```

| Paquete | Uso |
|---|---|
| `tkinter` | GUI (incluido en Python estándar) |
| `Pillow` | Renderizado del plano de fondo, calibración visual |
| `opencv-python` | Pipeline de detección de paredes |
| `numpy` | Operaciones matriciales en detección |

### Dependencias opcionales (para cargar PDFs)

```bash
# Opción A — recomendada, sin dependencias del sistema:
pip install pymupdf

# Opción B — requiere poppler instalado:
pip install pdf2image
sudo apt install poppler-utils
```

### Verificar instalación

```bash
python3 -c "import tkinter, PIL, cv2, numpy; print('Todo OK')"
```

---

## 3. Cómo ejecutar

Desde la raíz del repositorio `g1man`:

```bash
# Modo normal
python3 wallforge_studio/main.py

# Abriendo un proyecto existente directamente
python3 wallforge_studio/main.py mi_mapa.wfp
```

---

## 4. Interfaz de usuario

```
┌────────────────────────────────────────────────────────────────────┐
│  Menú: Archivo │ Editar │ Vista │ Herramientas │ Plano │ Exportar  │
├──────────┬─────────────────────────────────────────┬───────────────┤
│          │                                         │               │
│ TOOLBAR  │                                         │  PROPIEDADES  │
│          │          CANVAS PRINCIPAL               │               │
│ Selec.   │   (rejilla + plano de fondo + muros)    │  ID / long. / │
│ Dibujar  │                                         │  ángulo / mat │
│ Borrar   │                                         │               │
│          │                                         │  PROYECTO     │
│ Deshacer │                                         │  CONFIGURAC.  │
│ Rehacer  │                                         │               │
│ Eliminar │                                         │  PLANO FONDO  │
│          │                                         │  opacidad     │
│ Ajustar  │                                         │  detectar...  │
│ Centro   │                                         │               │
│          │                                         │               │
│ MuJoCo   │                                         │               │
│ g1man    │                                         │               │
├──────────┴─────────────────────────────────────────┴───────────────┤
│  Herramienta: Dibujar  │  (0.000, 0.000) m  │  12 muros           │
└────────────────────────────────────────────────────────────────────┘
```

### Canvas

- **Zoom**: rueda del ratón (rango: 4–600 px/m)
- **Pan**: botón central o botón derecho + arrastrar
- **Rejilla**: líneas menores (gris oscuro) y mayores (cada 5 celdas), eje X/Y resaltado
- **Etiquetas de distancia**: aparecen en las líneas de cuadrícula mayor

### Panel de propiedades (derecha)

Cuando hay un muro seleccionado muestra y permite editar:
- ID único, longitud, ángulo
- Grosor y altura (editables con "Aplicar cambios")
- Material MuJoCo
- Estado (`confirmed` / `detected` / `locked`)

La sección **PLANO DE FONDO** muestra nombre del archivo, dimensiones en píxeles y escala actual (px/m), con un slider de opacidad en tiempo real.

---

## 5. Herramientas de edición

### Seleccionar (`S`)

Clic sobre un muro para seleccionarlo (se muestra en rojo).  
`Shift + clic` para añadir/quitar muros de la selección.  
Clic en espacio vacío deselecciona todo.

**Edición de extremos (drag):**  
Cuando un muro está seleccionado aparecen **cuadrados naranjas** en sus dos extremos.  
Al pasar el cursor sobre uno el puntero cambia a `fleur`.  
Arrastrando el cuadrado se reposiciona ese extremo con snap activo:

- El snap de rejilla y de endpoints sigue funcionando durante el drag
- Con `Shift` se activa el snap de ángulo (múltiplos de 15°) tomando como referencia el extremo opuesto
- La barra de estado muestra coordenadas exactas y longitud actualizada en tiempo real
- Al soltar el botón se registra el estado en el histórico de deshacer
- `Esc` durante el drag cancela la operación y restaura la posición original

### Dibujar (`D`)

1. Primer clic fija el extremo inicial (aparece un punto naranja)
2. Mover el ratón muestra una línea de previsualización punteada
3. Segundo clic crea el muro
4. `Esc` cancela el muro en curso

Los nuevos muros heredan el grosor y altura configurados en el panel de propiedades.

### Borrar (`X`)

Clic sobre cualquier muro para eliminarlo inmediatamente.  
El muro bajo el cursor se resalta en verde (hover) antes de borrar.

### Historial de deshacer/rehacer

- `Ctrl+Z` — deshacer (hasta 60 pasos)
- `Ctrl+Y` o `Ctrl+Shift+Z` — rehacer

Cada operación que modifica el mapa (dibujar, borrar, drag de extremo, detección, confirmar) guarda un snapshot completo de todos los muros.

---

## 6. Sistema de snap

El snap se aplica en orden de **prioridad decreciente**:

| Prioridad | Tipo | Activación | Indicador en status bar |
|---|---|---|---|
| 1 | **Endpoint** | Siempre (si habilitado) | `⊙ endpoint` |
| 2 | **Grid** | Siempre (si habilitado) | `⊞ grid` |
| 3 | **Angle** | Solo con `Shift` + anchor | `∠ 15°` |

- **Endpoint snap**: engancha al extremo de cualquier muro existente dentro de la tolerancia configurada
- **Grid snap**: redondea al múltiplo más cercano del tamaño de rejilla (por defecto 0,5 m)
- **Angle snap**: mantiene la distancia y ajusta el ángulo al múltiplo de 15° más cercano respecto al punto ancla

Configuración en el panel derecho (sección CONFIGURACIÓN):

| Parámetro | Por defecto | Descripción |
|---|---|---|
| Rejilla (m) | 0,5 | Tamaño de celda de rejilla |
| Snap tol. (m) | 0,3 | Radio de enganche a extremos |

`Ctrl+G` activa/desactiva el snap completo.  
`G` activa/desactiva la visualización de la rejilla.

---

## 7. Plano de fondo (floor plan)

### Importar

**Menú Plano → Importar imagen / PDF… (`Ctrl+I`)** o botón "Importar plano…" en el panel derecho.

Formatos soportados: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP, PDF.

Tras la importación se abre automáticamente el diálogo de calibración.

### Calibración visual de escala

**Menú Plano → Calibrar escala (px/m)…**

La calibración muestra el plano en un canvas interactivo donde el usuario dibuja líneas de medida directamente sobre la imagen:

1. **1er clic** — fija el punto inicial (círculo naranja)
2. Mover el ratón — línea de previsualización punteada
3. **2º clic** — fija el punto final y abre un campo de entrada en el panel derecho
4. Escribir la **distancia real en metros** y pulsar `Enter` o "OK"
5. La medición queda registrada en la lista con su px/m calculado
6. Repetir el proceso con otra referencia conocida del plano para mayor precisión
7. El campo "Escala media" se actualiza automáticamente con el promedio de todas las mediciones
8. Pulsar **Aplicar** (o `Enter`) para guardar la escala en el proyecto

También es posible introducir el valor de px/m directamente si se conoce.  
Cada medición puede eliminarse individualmente con el botón `✕`.

### Controles del plano de fondo

| Control | Descripción |
|---|---|
| Opacidad (slider) | 0,05–1,0 en tiempo real |
| Mostrar plano | Activa/desactiva la visibilidad |
| Bloquear plano | Impide modificaciones accidentales |
| Eliminar plano | Quita el plano de fondo del proyecto |

El plano se renderiza eficientemente: solo se procesa y escala la porción visible del viewport. La opacidad se aplica mezclando la imagen con el color de fondo del canvas.

---

## 8. Detección automática de paredes

**Menú Plano → Detectar paredes del plano… (`Ctrl+D`)**

Requiere que haya un plano de fondo cargado y calibrado.

### Pipeline de detección

El sistema aplica 5 etapas sobre la imagen:

```
Imagen BGR
    │
    ▼
1. build_wall_mask()       — máscara binaria de píxeles oscuros (paredes)
    │                         combina Otsu + umbral HSV-V + umbral LAB-L
    │
    ▼
2. remove_text_noise()     — elimina componentes pequeños (texto, ruido)
    │                         filtra por área máxima y relación de aspecto
    │
    ▼
3. detect_hough()          — segmentos HoughLinesP sobre bordes Canny
   detect_components()     — rectángulos de componentes conectadas (minAreaRect)
    │
    ▼
4. merge_colinear()        — fusiona segmentos colineales cercanos en uno solo
    │                         criterio: ángulo similar + distancia normal pequeña
    │                                   + proyecciones solapadas/próximas
    │
    ▼
5. px_wall_to_world()      — conversión píxeles → metros (invierte eje Y)
    │
    ▼
List[Wall] en coordenadas mundo
```

### Parámetros ajustables en el diálogo

| Parámetro | Rango | Descripción |
|---|---|---|
| Umbral oscuridad | 20–200 | Umbral de valor/luminancia para considerar un píxel como pared |
| Blur (px) | 1–11 | Radio del suavizado Gaussiano pre-procesado |
| Cierre morfológico (px) | 0–15 | Cierre morfológico para conectar paredes discontinuas |
| Longitud mínima (px) | 10–200 | Longitud mínima de segmento Hough |
| Hueco máximo (px) | 0–50 | Hueco máximo permitido en segmentos Hough |
| Fusión (px) | 2–40 | Distancia máxima para fusionar segmentos colineales |
| Plano invertido | checkbox | Para planos con fondo negro y líneas claras |
| Filtrar ruido de texto | checkbox | Elimina componentes pequeñas (texto, cotas) |

### Flujo de trabajo recomendado

1. Importar plano → calibrar escala
2. Abrir diálogo de detección
3. Pulsar **Detectar** y observar el número de muros encontrados
4. Ajustar parámetros si el resultado no es satisfactorio y repetir
5. Pulsar **Añadir al mapa** — los muros se añaden con estado `detected` (color amarillo)
6. En el canvas, revisar los muros detectados
7. Pulsar **Confirmar muros detectados** (menú Plano o botón del panel) para convertirlos a `confirmed`
8. Usar la herramienta Seleccionar para ajustar extremos incorrectos

---

## 9. Exportación a MuJoCo XML

### Modo básico — escena genérica

**Menú Exportar → Exportar MuJoCo XML básico…**

Genera un archivo XML MuJoCo autocontenido, sin robot, con ángulos en **grados**:

```xml
<!-- WallForge Studio — 2025-05-08 12:00:00 -->
<mujoco model="wallforge_scene">
  <visual>...</visual>
  <asset>
    <material name="wall_mat" rgba="0.55 0.50 0.45 1" reflectance="0.05"/>
    ...
  </asset>
  <worldbody>
    <light .../>
    <geom name="floor" type="plane" .../>
    <geom name="Wall_0" type="box" pos="1.0000 2.0000 1.2500"
          size="1.5000 0.0750 1.2500" euler="0 0 0.0000"
          material="wall_mat" friction="1 0.05 0.01" group="3"/>
    ...
  </worldbody>
</mujoco>
```

Conversión segmento → geom box:

| Campo MJCF | Valor |
|---|---|
| `pos` | `(midx, midy, height/2)` — centro del geom |
| `size` | `(length/2, thickness/2, height/2)` — half-extents |
| `euler` | `"0 0 {ángulo_en_grados}"` |

### Modo repo g1man

**Menú Exportar → Exportar para repo g1man…**

Genera una escena compatible directamente con el repositorio g1man. Diferencias clave respecto al modo básico:

| Aspecto | Básico | Repo g1man |
|---|---|---|
| Ángulos euler | grados | **radianes** (por `<compiler angle="radian"/>` del robot) |
| Robot | no incluido | `<include file="g1_29dof.xml"/>` |
| `<statistic>` | no | `center="0 0 1.0" extent="{auto}"` |
| Nombre de muros | `Wall_0` configurable | `Wall_0`, `Wall_1`, … |
| Luces | altura 5 | altura 10 (escala de mapa) |

Tras exportar, copiar el archivo a `mujoco/simulacion/` del repositorio y ejecutar:

```bash
python3 mujoco/simulacion/run_sim_ai_g1.py
```

### Configuración de exportación (panel derecho)

| Parámetro | Por defecto | Descripción |
|---|---|---|
| Altura muro (m) | 2,5 | Altura de todas las paredes |
| Grosor muro (m) | 0,15 | Grosor por defecto de nuevas paredes |

Nota: solo se exportan muros con estado `confirmed` o `locked`. Los muros en estado `detected` (pendientes de validación) se excluyen.

---

## 10. Gestión de proyectos

### Nuevo proyecto

**`Ctrl+N`** — pide nombre y limpia el canvas.

### Guardar / Abrir

| Acción | Atajo |
|---|---|
| Guardar | `Ctrl+S` |
| Guardar como… | `Ctrl+Shift+S` |
| Abrir… | `Ctrl+O` |

Extensión nativa: `.wfp` (WallForge Project). También admite `.json`.

El título de la ventana muestra un `•` cuando hay cambios sin guardar.

---

## 11. Atajos de teclado

### Herramientas

| Tecla | Acción |
|---|---|
| `S` | Herramienta Seleccionar |
| `D` | Herramienta Dibujar |
| `X` | Herramienta Borrar |

### Edición

| Tecla | Acción |
|---|---|
| `Ctrl+Z` | Deshacer |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Rehacer |
| `Del` / `Backspace` | Eliminar muros seleccionados |
| `Ctrl+A` | Seleccionar todos los muros |
| `Esc` | Cancelar acción / deseleccionar |
| `Shift` (al dibujar/arrastrar) | Snap de ángulo (15°) |

### Vista

| Tecla | Acción |
|---|---|
| `F` | Ajustar vista al contenido |
| `Home` | Centrar en el origen (0, 0) |
| `G` | Mostrar/ocultar rejilla |
| `Ctrl+G` | Activar/desactivar snap |
| Rueda ratón | Zoom |
| Botón central/derecho + arrastrar | Pan |

### Archivo

| Tecla | Acción |
|---|---|
| `Ctrl+N` | Nuevo proyecto |
| `Ctrl+O` | Abrir proyecto |
| `Ctrl+S` | Guardar |
| `Ctrl+Shift+S` | Guardar como |
| `Ctrl+Q` | Salir |

### Plano de fondo

| Tecla | Acción |
|---|---|
| `Ctrl+I` | Importar imagen/PDF |
| `Ctrl+D` | Abrir diálogo de detección |

---

## 12. Arquitectura del código

```
wallforge_studio/
├── main.py                   # Punto de entrada, carga argumento de proyecto
│
├── model/
│   ├── wall.py               # Dataclass Wall + WallState (CONFIRMED/DETECTED/LOCKED)
│   └── project.py            # Dataclass Project + BackgroundLayer + ExportSettings
│
├── editor/
│   ├── canvas.py             # Widget WallCanvas: render, zoom/pan, dispatch de eventos
│   ├── tools.py              # BaseTool, SelectTool (drag extremos), DrawTool, DeleteTool
│   └── snap.py               # SnapEngine: endpoint > grid > angle
│
├── image/
│   ├── loader.py             # Carga PNG/JPG/BMP/TIFF/PDF → ndarray BGR
│   └── detector.py           # Pipeline completo: máscara → Hough+CC → fusión → Wall
│
├── export/
│   ├── mujoco_exporter.py    # Exportación básica (euler en grados)
│   └── repo_scene_exporter.py# Exportación g1man (euler en radianes, incluye robot)
│
├── ui/
│   └── app.py                # WallForgeApp: menú, toolbar, panel propiedades, diálogos
│
└── utils/
    └── geometry.py           # distance, snap_to_grid, snap_angle, closest_endpoint, ...
```

### Flujo de datos principal

```
WallForgeApp (ui/app.py)
    │  tiene un  Project (model/project.py)
    │  tiene un  WallCanvas (editor/canvas.py)
    │
    ├── WallCanvas despacha eventos a la herramienta activa (BaseTool)
    │     SelectTool  — selección + drag de extremos
    │     DrawTool    — creación de nuevos muros
    │     DeleteTool  — eliminación por clic
    │
    ├── SnapEngine se consulta en cada on_press / on_move del DrawTool y SelectTool
    │
    ├── BackgroundLayer (parte de Project)
    │     load_image()  →  _np_bgr (OpenCV) + _pil_rgb (Pillow)
    │     render via WallCanvas._draw_background()
    │
    ├── DetectionParams + detect_walls()  →  List[Wall] con state=DETECTED
    │
    └── export_basic() / export_repo_scene()  →  archivo .xml
```

---

## 13. Sistema de coordenadas

WallForge usa dos sistemas de coordenadas:

| Sistema | Eje X | Eje Y | Unidad |
|---|---|---|---|
| **Mundo** (muros, export) | derecha → | arriba ↑ | metros |
| **Pantalla** (canvas Tk) | derecha → | abajo ↓ | píxeles |

Transformaciones:

```
# Mundo → Pantalla
sx = cx + (wx - pan_x) * zoom
sy = cy - (wy - pan_y) * zoom

# Pantalla → Mundo
wx = (sx - cx) / zoom + pan_x
wy = -(sy - cy) / zoom + pan_y
```

donde `(cx, cy)` es el centro del canvas en píxeles.

Esta convención es coherente con MuJoCo, donde el plano del suelo es XY con Z hacia arriba.

Al convertir píxeles de imagen a coordenadas mundo, el eje Y se invierte:

```python
mx = origin_x + (px_col - img_w/2) / ppm
my = origin_y - (px_row - img_h/2) / ppm  # ← negativo
angle_world = -angle_image                  # ← negativo
```

---

## 14. Formato de archivo `.wfp`

JSON con la siguiente estructura:

```json
{
  "name": "mi_mapa",
  "grid_size": 0.5,
  "snap_enabled": true,
  "snap_tolerance": 0.3,
  "export": {
    "wall_height": 2.5,
    "wall_thickness": 0.15,
    "include_floor": true,
    "include_lights": true,
    "include_robot": false,
    "wall_prefix": "Wall",
    "world_name": "wallforge_scene",
    "robot_include": "g1_29dof.xml"
  },
  "walls": [
    {
      "id": "a3f1b2c4",
      "p1": [0.0, 0.0],
      "p2": [5.0, 0.0],
      "thickness": 0.15,
      "height": 2.5,
      "material": "wall_mat",
      "state": "confirmed"
    }
  ],
  "background": {
    "image_path": "/ruta/absoluta/plano.png",
    "ppm": 125.0,
    "origin_world": [0.0, 0.0],
    "opacity": 0.5,
    "visible": true,
    "locked": false
  }
}
```

- El campo `background` es opcional (solo presente si hay plano cargado).
- La imagen se referencia por ruta absoluta; si al abrir el proyecto la ruta no existe, se avisa al usuario pero el resto del proyecto se carga igualmente.
- Los datos de imagen (`_np_bgr`, `_pil_rgb`) no se serializan; se recargan desde disco al abrir.

---

## Notas de desarrollo

- **Python 3.8+** requerido. Probado en Python 3.10.
- No requiere `__init__.py` en los paquetes (namespace packages de Python 3.3+).
- El canvas usa `tkinter.Canvas` nativo; el renderizado del plano de fondo usa `PIL.ImageTk.PhotoImage`.
- La detección se ejecuta en el hilo principal (puede bloquear brevemente la UI con imágenes grandes). Para imágenes superiores a 4000×4000 px se recomienda reducir resolución antes de importar.
