# Crear ambiente virtual

conda create -n streamlit_pyspark python=3.10 -y

# Activar el ambiente

conda activate streamlit_pyspark

# Si tienes error de java

https://adoptium.net/es/download

# Ejemplo de ejecución 
streamlit run hello.py

# Instalar dependencias

pip install streamlit pyspark pandas plotly scikit-learn openpyxl

```

### 1.2 Estructura del Proyecto

Vamos a crear esta estructura de carpetas:
```

proyecto_ventas_autos/
│
├── app.py # Aplicación principal de Streamlit
├── data/ # Carpeta para datasets
│ └── car_sales.csv # Dataset de ventas de autos
├── models/ # Modelos entrenados
└── utils/ # Funciones auxiliares
├── **init**.py
├── data_processing.py # Procesamiento de datos
└── visualization.py # Funciones de visualización

