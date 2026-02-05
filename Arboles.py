import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
warnings.filterwarnings('ignore')

# configs basicas para las graficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# funcion para crear los datos randoms de estudiantes
def crear_datos_estudiantes(cant_datos=1000):
    np.random.seed(20)  # para que siempre salgan los mismos datos
    
    carreras = ['Mecatrónica', 'Electrónica', 'Sistemas', 'Industrial']
    
    # creamos un diccionario con los datos
    datos = {
        'Carrera': np.random.choice(carreras, cant_datos),
        'Horas_Estudio': np.random.normal(25, 10, cant_datos),
        'Promedio_Acumulado': np.random.normal(3.7, 0.5, cant_datos),
        'Asistencia': np.random.normal(80, 15, cant_datos)
    }
    
    df = pd.DataFrame(datos)
    
    # ajustar los rangos para que sean realistas
    df['Horas_Estudio'] = df['Horas_Estudio'].clip(0, 60)
    df['Promedio_Acumulado'] = df['Promedio_Acumulado'].clip(2.5, 5.0)
    df['Asistencia'] = df['Asistencia'].clip(40, 100)
    
    # calcular el desempeño basado en las otras variables
    desempeños = []
    for i in range(len(df)):
        # hacemos un score con las variables
        puntaje = (df.iloc[i]['Horas_Estudio'] * 0.4 + 
                   (df.iloc[i]['Promedio_Acumulado'] - 2.5) * 20 + 
                   df.iloc[i]['Asistencia'] * 0.3)
        
        # le agregamos algo de ruido random
        puntaje += np.random.normal(0, 5)
        
        # clasificamos segun el puntaje
        if puntaje < 50:
            desempeños.append('Bajo')
        elif puntaje < 70:
            desempeños.append('Medio')
        else:
            desempeños.append('Alto')
    
    df['Desempeño'] = desempeños
    
    return df


# preparar los datos para entrenar
def preparar_datos(df):
    # convertir la carrera a numeros (one-hot encoding)
    df_numerico = pd.get_dummies(df, columns=['Carrera'], prefix='Carrera')
    
    # separar las X (caracteristicas) de las y (lo que queremos predecir)
    X = df_numerico.drop('Desempeño', axis=1)
    y = df_numerico['Desempeño']
    
    # dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20, stratify=y
    )
    
    print("Datos listos:")
    print(f"  Total: {len(df)} estudiantes")
    print(f"  Para entrenar: {len(X_train)}")
    print(f"  Para probar: {len(X_test)}")
    print()
    
    return X_train, X_test, y_train, y_test, X.columns


# entrenar con diferentes cantidades de arboles
def entrenar_modelos(X_train, X_test, y_train, y_test):
    print("=" * 60)
    print("Entrenando con diferentes números de árboles...")
    print("=" * 60)
    
    num_arboles = [1, 5, 10, 20, 50, 100]
    resultados = []
    modelopast = None
    modelo_final = None
    exactitud_train_past = 0.0

    for n in num_arboles:
        # crear el modelo
        modelo = RandomForestClassifier(n_estimators=n, random_state=20, n_jobs=-1)
        modelo.fit(X_train, y_train)
        
        # hacer predicciones
        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)
        
        # calcular exactitud
        exactitud_train = accuracy_score(y_train, pred_train)
        exactitud_test = accuracy_score(y_test, pred_test)
        
        resultados.append({
            'arboles': n,
            'exactitud_entrenamiento': exactitud_train,
            'exactitud_prueba': exactitud_test
        })
        
        print(f"\nCon {n} árboles:")
        print(f"  Exactitud entrenamiento: {exactitud_train:.4f}")
        print(f"  Exactitud prueba: {exactitud_test:.4f}")
        
        modelopast = modelo
        if n == 1: 
            exactitud_train_past = exactitud_train
            print("El mejor número de arboles por ahora es: " + str(n))
        # guardamos el modelo de 100 arboles
        if n > 1 and exactitud_train_past + .01 < exactitud_train: 
            exactitud_train_past = exactitud_train
            modelo_final = modelo
            print("El mejor número de arbol por ahora es: " + str(n))
          
    
    print("\n" + "=" * 60)
    return modelo_final, resultados


# ver que variables son mas importantes
def analizar_importancias(modelo, nombres_columnas):
    importancias = modelo.feature_importances_
    
    # crear tabla ordenada
    datos_import = pd.DataFrame({
        'Variable': nombres_columnas,
        'Importancia': importancias
    }).sort_values('Importancia', ascending=False)
    
    print("\n" + "=" * 60)
    print("Importancia de cada variable:")
    print("=" * 60)
    print(datos_import.to_string(index=False))
    print()
    
    return datos_import


# mostrar reporte final
def mostrar_reporte(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    
    print("\n" + "=" * 60)
    print("Reporte del modelo final:")
    print("=" * 60)
    print(classification_report(y_test, predicciones))





#LATONERIA Y PINTURA PARA LA INTERFAZ 

# crear las graficas
def hacer_graficas(df, resultados, modelo, nombres_cols, X_test, y_test):
    fig, ejes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis del Random Forest', fontsize=16, fontweight='bold')
    
    # grafica 1: exactitud vs numero de arboles
    df_res = pd.DataFrame(resultados)
    ejes[0, 0].plot(df_res['arboles'], df_res['exactitud_entrenamiento'], 
                   marker='o', label='Entrenamiento', linewidth=2)
    ejes[0, 0].plot(df_res['arboles'], df_res['exactitud_prueba'], 
                   marker='s', label='Prueba', linewidth=2)
    ejes[0, 0].set_xlabel('Número de Árboles')
    ejes[0, 0].set_ylabel('Exactitud')
    ejes[0, 0].set_title('Exactitud vs Número de Árboles')
    ejes[0, 0].legend()
    ejes[0, 0].grid(True, alpha=0.3)
    
    # grafica 2: importancia de variables
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    ejes[0, 1].barh(range(len(importancias)), importancias[indices])
    ejes[0, 1].set_yticks(range(len(importancias)))
    ejes[0, 1].set_yticklabels([nombres_cols[i] for i in indices])
    ejes[0, 1].set_xlabel('Importancia')
    ejes[0, 1].set_title('Importancia de Variables')
    ejes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # grafica 3: matriz de confusion
    predicciones = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, predicciones)
    
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ejes[1, 0],
               xticklabels=['Alto', 'Bajo', 'Medio'],
               yticklabels=['Alto', 'Bajo', 'Medio'])
    ejes[1, 0].set_xlabel('Predicción')
    ejes[1, 0].set_ylabel('Real')
    ejes[1, 0].set_title('Matriz de Confusión')
    
    # grafica 4: distribucion de desempeño
    conteo = df['Desempeño'].value_counts()
    colores = ['#2ecc71', '#e74c3c', '#f39c12']
    ejes[1, 1].pie(conteo.values, labels=conteo.index,
                  autopct='%1.1f%%', colors=colores, startangle=90)
    ejes[1, 1].set_title('Distribución de Desempeño')
    
    plt.tight_layout()
    plt.savefig('analisis_random_forest.png', dpi=300, bbox_inches='tight')
    print("Gráficas guardadas en 'analisis_random_forest.png'")
    plt.show()


# interfaz grafica
class Interfaz:
    def __init__(self, modelo, columnas):
        self.modelo = modelo
        self.columnas = columnas
        self.ventana = tk.Tk()
        self.setup_ventana()
        self.crear_interfaz()
        
    def setup_ventana(self):
        self.ventana.title("Predictor de Desempeño - Random Forest")
        self.ventana.geometry("600x700")
        self.ventana.resizable(False, False)
        self.ventana.configure(bg='#2c3e50')
        
    def crear_interfaz(self):
        # titulo
        frame_titulo = tk.Frame(self.ventana, bg='#34495e', padx=20, pady=15)
        frame_titulo.pack(fill='x')
        
        tk.Label(frame_titulo, text="🎓 Predictor de Desempeño Académico",
                font=('Arial', 18, 'bold'), bg='#34495e', fg='white').pack()
        
        tk.Label(frame_titulo, text="Random Forest - UMNG",
                font=('Arial', 11), bg='#34495e', fg='#bdc3c7').pack()
        
        # formulario
        frame_form = tk.Frame(self.ventana, bg='#2c3e50', padx=30, pady=20)
        frame_form.pack(fill='both', expand=True)
        
        # campo carrera
        tk.Label(frame_form, text="Carrera:", font=('Arial', 12, 'bold'),
                bg='#2c3e50', fg='white').grid(row=0, column=0, sticky='w', pady=10)
        
        self.carrera = tk.StringVar()
        combo = ttk.Combobox(frame_form, textvariable=self.carrera,
                            values=['Mecatrónica', 'Electrónica', 'Sistemas', 'Industrial'],
                            state='readonly', width=30, font=('Arial', 11))
        combo.grid(row=0, column=1, pady=10, padx=10)
        combo.current(0)
        
        # campo horas de estudio
        tk.Label(frame_form, text="Horas de Estudio (semanal):",
                font=('Arial', 12, 'bold'), bg='#2c3e50', fg='white').grid(
                    row=1, column=0, sticky='w', pady=10)
        
        self.horas = tk.Entry(frame_form, width=32, font=('Arial', 11))
        self.horas.grid(row=1, column=1, pady=10, padx=10)
        self.horas.insert(0, "25")
        
        tk.Label(frame_form, text="(Entre 0 y 60 horas)", font=('Arial', 9, 'italic'),
                bg='#2c3e50', fg='#95a5a6').grid(row=2, column=1, sticky='w', padx=10)
        
        # campo promedio
        tk.Label(frame_form, text="Promedio Acumulado:", font=('Arial', 12, 'bold'),
                bg='#2c3e50', fg='white').grid(row=3, column=0, sticky='w', pady=10)
        
        self.promedio = tk.Entry(frame_form, width=32, font=('Arial', 11))
        self.promedio.grid(row=3, column=1, pady=10, padx=10)
        self.promedio.insert(0, "3.8")
        
        tk.Label(frame_form, text="(Entre 2.5 y 5.0)", font=('Arial', 9, 'italic'),
                bg='#2c3e50', fg='#95a5a6').grid(row=4, column=1, sticky='w', padx=10)
        
        # campo asistencia
        tk.Label(frame_form, text="Asistencia (%):", font=('Arial', 12, 'bold'),
                bg='#2c3e50', fg='white').grid(row=5, column=0, sticky='w', pady=10)
        
        self.asistencia = tk.Entry(frame_form, width=32, font=('Arial', 11))
        self.asistencia.grid(row=5, column=1, pady=10, padx=10)
        self.asistencia.insert(0, "85")
        
        tk.Label(frame_form, text="(Entre 40% y 100%)", font=('Arial', 9, 'italic'),
                bg='#2c3e50', fg='#95a5a6').grid(row=6, column=1, sticky='w', padx=10)
        
        # boton predecir
        frame_btns = tk.Frame(frame_form, bg='#2c3e50')
        frame_btns.grid(row=7, column=0, columnspan=2, pady=30)
        
        tk.Button(frame_btns, text="🔮 Predecir Desempeño", command=self.predecir,
                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                 width=25, height=2, cursor='hand2', relief='raised', bd=3
                 ).pack()
        
        # label para mostrar resultado
        self.resultado = tk.Label(frame_form, text="", font=('Arial', 14, 'bold'),
                                 bg='#2c3e50', fg='#ecf0f1', wraplength=500)
        self.resultado.grid(row=8, column=0, columnspan=2, pady=20)
        
        # creditos
        frame_cred = tk.Frame(self.ventana, bg='#34495e', pady=10)
        frame_cred.pack(fill='x', side='bottom')
        
        tk.Label(frame_cred, 
                text="v1.0 | Universidad Militar Nueva Granada\nLab de IA | 2026",
                font=('Arial', 9), bg='#34495e', fg='#bdc3c7').pack()
        
    def predecir(self):
        try:
            # leer valores
            carrera = self.carrera.get()
            hrs = float(self.horas.get())
            prom = float(self.promedio.get())
            asist = float(self.asistencia.get())
            
            # validar
            if not (0 <= hrs <= 60):
                raise ValueError("Las horas deben estar entre 0 y 60")
            if not (2.5 <= prom <= 5.0):
                raise ValueError("El promedio debe estar entre 2.5 y 5.0")
            if not (40 <= asist <= 100):
                raise ValueError("La asistencia debe estar entre 40% y 100%")
            
            # preparar datos para predecir
            datos_input = pd.DataFrame({
                'Horas_Estudio': [hrs],
                'Promedio_Acumulado': [prom],
                'Asistencia': [asist],
                'Carrera_Electrónica': [1 if carrera == 'Electrónica' else 0],
                'Carrera_Industrial': [1 if carrera == 'Industrial' else 0],
                'Carrera_Mecatrónica': [1 if carrera == 'Mecatrónica' else 0],
                'Carrera_Sistemas': [1 if carrera == 'Sistemas' else 0]
            })
            
            datos_input = datos_input[self.columnas]
            
            # hacer prediccion
            pred = self.modelo.predict(datos_input)[0]
            probs = self.modelo.predict_proba(datos_input)[0]
            
            # calcular confianza
            clases = self.modelo.classes_
            idx = list(clases).index(pred)
            conf = probs[idx] * 100
            
            # mostrar resultado
            if pred == 'Alto':
                color = '#27ae60'
                emoji = '🌟'
            elif pred == 'Medio':
                color = '#f39c12'
                emoji = '⭐'
            else:
                color = '#e74c3c'
                emoji = '⚠️'
            
            self.resultado.config(
                text=f"{emoji} Desempeño: {pred}\nConfianza: {conf:.1f}%",
                fg=color
            )
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Algo salió mal:\n{str(e)}")
    
    def iniciar(self):
        self.ventana.mainloop()










# funcion principal CODIFICACION 
def main():
    print("=" * 70)
    print("         LABORATORIO DE RANDOM FOREST")
    print("      Predicción de Desempeño Académico")
    print("    Universidad Militar Nueva Granada")
    print("=" * 70)
    print()
    
    # paso 1: crear datos
    print("PASO 1: Creando datos sintéticos de estudiantes...")
    print("-" * 70)
    datos = crear_datos_estudiantes(1000)
    print(f"Dataset creado con {len(datos)} estudiantes")
    print("\nPrimeros datos:")
    print(datos.head(10))
    print("\nDistribución:")
    print(datos['Desempeño'].value_counts())
    print()
    
    # paso 2: entrenar modelo
    print("\nPASO 2: Entrenando el modelo...")
    print("-" * 70)
    X_train, X_test, y_train, y_test, cols = preparar_datos(datos)
    modelo, resultados = entrenar_modelos(X_train, X_test, y_train, y_test)
    
    # paso 3: analizar
    print("\nPASO 3: Analizando resultados...")
    print("-" * 70)
    analizar_importancias(modelo, cols)
    mostrar_reporte(modelo, X_test, y_test)
    
    print("\nCreando gráficas...")
    hacer_graficas(datos, resultados, modelo, cols, X_test, y_test)
    
    # paso 4: interfaz
    print("\nPASO 4: Abriendo interfaz...")
    print("-" * 70)
    print("Abriendo ventana de predicción...")
    print("(Cierra la ventana cuando termines)")
    print()
    
    app = Interfaz(modelo, cols)
    app.iniciar()
    
    print("\n" + "=" * 70)
    print("            LABORATORIO TERMINADO")
    print("=" * 70)


if __name__ == "__main__":
    main()