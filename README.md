# hemolytic-activity-ml-prediction

## Estudiante: Samuel Salazar Diaz

Predicción de actividad hemolítica en péptidos bioactivos mediante descriptores fisicoquímicos y modelos de machine learning, con análisis e interpretación de las variables más influyentes.

Trabajo desarrollado para la **Actividad 1** del Programa Superior Universitario en Bioinformática Aplicada al Desarrollo de Fármacos: *Modelado predictivo de actividad hemolítica mediante aprendizaje automático*. Está basado en los conjuntos de datos **HemoPI-1, HemoPI-2 y HemoPI-3** del artículo de Plisson et al. (2020).

## Resumen del flujo de trabajo

Todo el análisis se ejecutó en **Google Colab** vinculado a **Google Drive**, en tres notebooks consecutivos. Cada notebook lee las salidas del anterior.

```mermaid
%%{init: {"fontFamily": "Arial, Helvetica, sans-serif", "flowchart": {"htmlLabels": false, "padding": 20, "nodeSpacing": 40, "rankSpacing": 50}}}%%
graph TD
    A["Datasets HemoPI-1, 2 y 3<br/>12 archivos FASTA"] --> B

    subgraph P1["Parte 1: Curación de datos"]
        B["Lectura de FASTA<br/>id, secuencia y etiqueta"] --> C["Curación de datos<br/>faltantes, no estándar,<br/>contradictorias y duplicados"]
        C --> D["Control de fuga de datos<br/>entrenamiento vs validación"]
    end

    D --> E["Entrenamiento: archivos model<br/>Validación: archivos validation"]
    E --> F

    subgraph P2["Parte 2: SMILES y descriptores"]
        F["Secuencia a SMILES<br/>pyPept (vía BILN)"] --> G["45 descriptores 2D<br/>Mordred"]
        G --> H["Limpieza con entrenamiento<br/>NaN, constantes, r > 0,9"]
    end

    H --> I

    subgraph P3["Parte 3: Modelos de ML"]
        I["Pipeline por modelo<br/>selección de variables +<br/>GridSearchCV (CV de 5)"] --> J["Regresión logística<br/>+ RFE"]
        I --> K["Random Forest<br/>+ importancia"]
        I --> L["XGBoost<br/>+ importancia"]
        J --> M["Evaluación<br/>entrenamiento, CV y externa"]
        K --> M
        L --> M
        M --> N["Curvas ROC"]
        M --> O["Top 10 descriptores<br/>y correlaciones"]
    end

    N --> Q["Interpretación<br/>y limitaciones"]
    O --> Q
```

## Datos

Los datasets provienen del repositorio de GitHub asociado al artículo de Plisson et al. (2020), indicado en su sección de disponibilidad de datos. Están en la carpeta `datos/`. Cada dataset contiene 4 archivos FASTA:

| Archivo | Contenido | Uso en este trabajo |
|---|---|---|
| `HemoPI-X_model_class0.fasta` | Péptidos de clase 0 | Entrenamiento |
| `HemoPI-X_model_class1.fasta` | Péptidos de clase 1 | Entrenamiento |
| `HemoPI-X_validation_class0.fasta` | Péptidos de clase 0 | Validación externa (20 %) |
| `HemoPI-X_validation_class1.fasta` | Péptidos de clase 1 | Validación externa (20 %) |

En total son 12 archivos (3 datasets × 4). `class0` se codifica como `label = 0` y `class1` como `label = 1`. En el artículo, HemoPI-1 distingue péptidos hemolíticos de no hemolíticos, mientras que HemoPI-2 y HemoPI-3 se usan para predecir potencia hemolítica alta o baja. Los tres datasets se analizan de forma **independiente**.

## Estructura del repositorio

```
hemolytic-activity-ml-prediction/
├── README.md
├── LICENSE
├── 1_Curacion_HemoPI.ipynb      # Parte 1: recolección y curación
├── 2_SMILES_Mordred.ipynb       # Parte 2: SMILES y descriptores
├── 3_Modelos_ML.ipynb           # Parte 3: modelos de machine learning
├── datos/                       # datos originales (4 FASTA por dataset)
│   ├── HemoPI-1/
│   ├── HemoPI-2/
│   └── HemoPI-3/
├── datos_procesados/            # salida de la Parte 1
├── descriptores/                # salida de la Parte 2
└── resultados_modelos/          # salida de la Parte 3
    ├── modelos/
    ├── tablas/
    └── figuras/
```

La raíz del repositorio equivale a la carpeta `Act1_PSU` de Google Drive, donde se ejecutaron los notebooks.

## Descripción de los notebooks

### Parte 1: `1_Curacion_HemoPI.ipynb`
Recolección y curación de los datos peptídicos.

1. Monta Google Drive y comprueba que existan los 12 archivos FASTA.
2. Lee los FASTA y construye una tabla con `dataset`, `split`, `peptide_id` (línea `>`), `sequence`, `label` y `activity`.
3. Une los archivos `model` (clase 0 y 1) como **entrenamiento** y los `validation` como **validación externa**, por separado para cada dataset.
4. Cura los datos, con un registro de las filas eliminadas en cada paso:
   - valores faltantes o secuencias vacías;
   - secuencias con aminoácidos no estándar;
   - secuencias con etiquetas contradictorias;
   - duplicados exactos;
   - secuencias presentes a la vez en entrenamiento y validación (se quitan del entrenamiento para evitar fuga de datos).
5. Genera gráficos de balance de clases y de longitud de secuencias.

**Salidas** (`datos_procesados/`): `HemoPI-X_train.csv`, `HemoPI-X_validation.csv` y `registro_curacion.csv`.

### Parte 2: `2_SMILES_Mordred.ipynb`
Conversión de secuencias a SMILES y cálculo de descriptores fisicoquímicos.

1. Convierte cada secuencia a **SMILES** con **pyPept** (secuencia → BILN → molécula RDKit → SMILES). Incluye un parche de compatibilidad con versiones recientes de pandas.
2. Calcula con **Mordred** un subconjunto de **45 descriptores 2D**, agrupados en seis categorías: tamaño y composición, lipofilia y refractividad, polaridad y puentes de hidrógeno, ionizables y carga parcial, topología y forma, y polarizabilidad. Se seleccionaron a criterio porque calcular los 1613 descriptores de Mordred resultaba muy lento y pesado.
3. Limpia los descriptores, ajustando todo **solo con el entrenamiento**: se eliminan los descriptores con más de 5 % de valores faltantes, los constantes y los de alta correlación (|r| > 0,9). Se protegen `MW`, `SLogP`, `nHBDon` y `nHBAcc`, equivalentes en Mordred de `MolWt`, `MolLogP`, `NumHDonors` y `NumHAcceptors`.

**Salidas** (`descriptores/`):
- `HemoPI-X_train_smiles.csv` y `HemoPI-X_validation_smiles.csv`;
- `HemoPI-X_train_descriptores_crudo.csv` y `HemoPI-X_valid_descriptores_crudo.csv`, que son la caché del cálculo;
- `HemoPI-X_train_descriptores.csv` y `HemoPI-X_validation_descriptores.csv`, que son las tablas finales;
- `HemoPI-X_descriptores_seleccionados.txt` y `resumen_filtrado_descriptores.csv`.

### Parte 3: `3_Modelos_ML.ipynb`
Construcción, evaluación e interpretación de los modelos.

1. Entrena **3 algoritmos** en cada dataset, con selección de variables dentro de un `Pipeline`:

   | Modelo | Selección de variables |
   |---|---|
   | Regresión logística (con escalado) | RFE (eliminación recursiva) |
   | Random Forest | Selección por importancia (`SelectFromModel`) |
   | XGBoost | Selección por importancia (`SelectFromModel`) |

2. Ajusta hiperparámetros y el número de descriptores con `GridSearchCV` y validación cruzada estratificada de 5 pliegues (refit por AUC-ROC).
3. Evalúa en tres niveles: entrenamiento, validación cruzada y validación externa (20 %). Métricas: accuracy, precision, recall, especificidad, F1, AUC-ROC, MCC y kappa de Cohen, con intervalos bootstrap para AUC y MCC en la validación externa.
4. Genera las curvas ROC, el top 10 de descriptores más importantes por modelo y el análisis de correlaciones entre los descriptores seleccionados.
5. Guarda el modelo final y los pesos o importancias de los descriptores.

**Salidas** (`resultados_modelos/`):
- `modelos/`: `HemoPI-X_<modelo>_modelo_final.joblib` y `HemoPI-X_<modelo>_gridsearch.joblib`.
- `tablas/`: métricas, hiperparámetros, importancias, descriptores consistentes entre modelos y pares correlacionados.
- `figuras/`: `curvas_roc.png`, `top10_importancias.png` y `correlacion_HemoPI-X.png`.

## Cómo ejecutarlo

Los notebooks se ejecutaron en **Google Colab** con la carpeta `Act1_PSU` almacenada en Google Drive (`/content/drive/MyDrive/Act1_PSU`).

1. Sube a tu Google Drive el contenido del repositorio en una carpeta llamada `Act1_PSU`, en la raíz de *Mi unidad*.
2. Los notebooks esperan los FASTA originales directamente en `Act1_PSU/HemoPI-X/`. En el repositorio están dentro de `datos/`. Mueve las carpetas `HemoPI-1`, `HemoPI-2` y `HemoPI-3` a la raíz de `Act1_PSU`, o cambia en `1_Curacion_HemoPI.ipynb` la ruta `{BASE_DIR}/{ds}/` por `{BASE_DIR}/datos/{ds}/` (aparece en dos celdas).
3. Abre cada notebook en Colab (*Archivo → Subir cuaderno*) y ejecútalos **en orden**: `1_`, `2_` y `3_`.
4. Acepta el montaje de Drive cuando Colab lo solicite.
5. Si guardas la carpeta en otra ruta, cambia la variable `BASE_DIR` en la primera celda de configuración de cada notebook.

Los cálculos más lentos (descriptores de la Parte 2 y `GridSearchCV` de la Parte 3) se guardan en Drive. Si se corta la sesión, al volver a ejecutar se cargan sin recalcular. Para forzar el reentrenamiento de la Parte 3, pon `RECALCULAR = True`.

## Requerimientos

Entorno: Google Colab (Python 3). Las librerías se instalan en los propios notebooks.

| Librería | Uso |
|---|---|
| `pandas`, `numpy` | Manejo de datos |
| `matplotlib`, `seaborn` | Gráficos |
| `rdkit` | Manejo de moléculas y SMILES |
| `pyPept` | Conversión de secuencias a SMILES (se instala desde GitHub: `pip install git+https://github.com/Boehringer-Ingelheim/pyPept.git`) |
| `mordredcommunity` | Descriptores moleculares (fork mantenido de Mordred, se importa como `mordred`) |
| `scikit-learn` | Modelos, validación cruzada, selección de variables y métricas |
| `xgboost` | Modelo XGBoost |
| `joblib` | Guardado de modelos |
| `tqdm` | Barras de progreso |

> Versiones: completar con las del entorno de Colab usado, que se pueden obtener con `!pip list | grep -i -E "pandas|numpy|scikit-learn|xgboost|rdkit|mordred|pypept|seaborn|matplotlib"`.

## Resultados

Desempeño en la **validación externa (20 %)** de los modelos finales, con umbral de clasificación de 0,5. Entre corchetes, el intervalo de confianza del 95 % (*bootstrap*, 1000 remuestreos). En negrita, el mayor AUC-ROC y MCC de cada dataset.

| Dataset | Modelo | Accuracy | Precision | Recall | Especificidad | F1 | AUC-ROC [IC 95 %] | MCC [IC 95 %] | Kappa |
|---|---|---|---|---|---|---|---|---|---|
| HemoPI-1 | Regresión logística | 0,882 | 0,889 | 0,873 | 0,891 | 0,881 | 0,960 [0,936; 0,981] | 0,764 [0,673; 0,846] | 0,764 |
| HemoPI-1 | Random Forest | 0,936 | 0,962 | 0,909 | 0,964 | 0,935 | **0,989** [0,978; 0,997] | **0,874** [0,811; 0,936] | 0,873 |
| HemoPI-1 | XGBoost | 0,927 | 0,943 | 0,909 | 0,945 | 0,926 | 0,983 [0,967; 0,994] | 0,855 [0,786; 0,919] | 0,855 |
| HemoPI-2 | Regresión logística | 0,713 | 0,741 | 0,727 | 0,696 | 0,734 | 0,764 [0,699; 0,833] | 0,422 [0,307; 0,555] | 0,422 |
| HemoPI-2 | Random Forest | 0,762 | 0,782 | 0,782 | 0,739 | 0,782 | **0,856** [0,796; 0,908] | **0,521** [0,396; 0,644] | 0,521 |
| HemoPI-2 | XGBoost | 0,757 | 0,770 | 0,791 | 0,717 | 0,780 | 0,830 [0,773; 0,885] | 0,510 [0,389; 0,633] | 0,510 |
| HemoPI-3 | Regresión logística | 0,729 | 0,754 | 0,746 | 0,709 | 0,750 | 0,791 [0,741; 0,839] | 0,455 [0,352; 0,551] | 0,455 |
| HemoPI-3 | Random Forest | 0,794 | 0,789 | 0,847 | 0,730 | 0,817 | **0,859** [0,820; 0,898] | **0,583** [0,497; 0,671] | 0,581 |
| HemoPI-3 | XGBoost | 0,748 | 0,778 | 0,751 | 0,743 | 0,764 | 0,825 [0,779; 0,869] | 0,493 [0,391; 0,587] | 0,493 |

**Observaciones**

- **Random Forest obtuvo el mayor AUC-ROC y MCC en los tres datasets.** Sin embargo, sus intervalos de confianza se solapan con los de XGBoost, por lo que la diferencia entre ambos no es concluyente.
- **La regresión logística fue el modelo con menor desempeño en los tres datasets**, aunque en HemoPI-1 se mantiene alta (AUC-ROC de 0,960).
- **El desempeño es claramente mayor en HemoPI-1** (AUC-ROC entre 0,960 y 0,989) que en HemoPI-2 y HemoPI-3 (entre 0,764 y 0,859).

Las métricas de entrenamiento y de validación cruzada, la comparación interno-externo, las importancias de descriptores y las curvas ROC están en `resultados_modelos/tablas/` y `resultados_modelos/figuras/`.

## Limitaciones

- Los descriptores son **2D** (calculados desde el SMILES): no codifican el orden de la secuencia ni la estructura secundaria.
- Se usó un subconjunto de 45 descriptores elegidos a criterio, y el filtro de correlación conserva el primer descriptor de cada grupo redundante, no el más informativo.
- La validación externa es una partición aleatoria de la misma fuente: péptidos de secuencia parecida pueden quedar a ambos lados del corte.
- El tamaño de muestra es moderado, por lo que las diferencias pequeñas entre modelos pueden no ser significativas.
- La importancia de variables indica asociación estadística, no causalidad.

## Licencia

Ver el archivo [LICENSE](LICENSE).

## Referencias

- Plisson, F., Ramírez-Sánchez, O., & Martínez-Hernández, C. (2020). Machine learning-guided discovery and design of non-hemolytic peptides. *Scientific Reports, 10*, 16581. https://doi.org/10.1038/s41598-020-73644-6
- Ochoa, R., Brown, J. B., & Fox, T. (2023). pyPept: a python library to generate atomistic 2D and 3D representations of peptides. *Journal of Cheminformatics, 15*, 79. https://doi.org/10.1186/s13321-023-00748-2
- Moriwaki, H., Tian, Y.-S., Kawashita, N., & Takagi, T. (2018). Mordred: a molecular descriptor calculator. *Journal of Cheminformatics, 10*, 4. https://doi.org/10.1186/s13321-018-0258-y
