import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Función del problema
# f(x) = x^3 + x^2 - 1
# ---------------------------------------------------------
def f(x):
    return x**3 + x**2 - 1


# ---------------------------------------------------------
# Método de Falsa Posición ESTÁNDAR (Regula Falsi)
# ---------------------------------------------------------
def falsa_posicion_estandar(a, b, tolerancia=1e-5, max_iteraciones=100):
    """
    Método de Falsa Posición estándar.
    
    Problema: Presenta estancamiento porque uno de los extremos
    permanece fijo durante muchas iteraciones, ralentizando la convergencia.
    """
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, f(a), [], 0
    if fb == 0:
        return b, f(b), [], 0
    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido: f(a) y f(b) deben tener signos opuestos."
        )

    tabla = []
    r_anterior = None

    for iteracion in range(1, max_iteraciones + 1):
        # Fórmula de la falsa posición
        r = a - fa * (b - a) / (fb - fa)
        fr = f(r)

        if r_anterior is not None:
            error = abs(r - r_anterior)
        else:
            error = abs(b - a)
        
        r_anterior = r

        tabla.append({
            "iteracion": iteracion,
            "a": a,
            "b": b,
            "r": r,
            "f(r)": fr,
            "error": error,
            "metodo": "Estándar"
        })

        if abs(fr) < tolerancia or error < tolerancia:
            return r, fr, tabla, iteracion

        if fa * fr < 0:
            b = r
            fb = fr
        else:
            a = r
            fa = fr

    r = a - fa * (b - a) / (fb - fa)
    return r, f(r), tabla, max_iteraciones


# ---------------------------------------------------------
# Método de Falsa Posición MODIFICADO (Illinois Method)
# ---------------------------------------------------------
def falsa_posicion_illinois(a, b, tolerancia=1e-5, max_iteraciones=100):
    """
    Illinois Method (Falsa Posición Modificada)
    
    SOLUCIÓN AL ESTANCAMIENTO:
    
    El estancamiento ocurre porque cuando uno de los extremos (ej: 'b')
    permanece fijo y siempre tiene el mismo signo que f(r), el factor
    de peso (fb) no cambia, ralentizando la convergencia.
    
    La solución: Cuando un extremo se repite en dos iteraciones consecutivas,
    se reduce su valor funcional por un factor (típicamente 0.5):
    
    fb = fb / 2  (si 'b' es el extremo fijo)
    fa = fa / 2  (si 'a' es el extremo fijo)
    
    Esto reduce efectivamente el peso del extremo fijo en la fórmula,
    haciendo que la interpolación converja más rápidamente.
    """
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, f(a), [], 0
    if fb == 0:
        return b, f(b), [], 0
    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido: f(a) y f(b) deben tener signos opuestos."
        )

    tabla = []
    r_anterior = None
    a_anterior = None
    b_anterior = None

    for iteracion in range(1, max_iteraciones + 1):
        # Detección de estancamiento: reducir el factor del extremo fijo
        if a_anterior == a and fa * f(a - fa * (b - a) / (fb - fa)) > 0:
            # El extremo 'a' está fijo y sigue siendo el mismo lado del cero
            fa = fa / 2
        
        if b_anterior == b and fb * f(a - fa * (b - a) / (fb - fa)) > 0:
            # El extremo 'b' está fijo y sigue siendo el mismo lado del cero
            fb = fb / 2

        a_anterior = a
        b_anterior = b

        # Fórmula de falsa posición modificada
        r = a - fa * (b - a) / (fb - fa)
        fr = f(r)

        if r_anterior is not None:
            error = abs(r - r_anterior)
        else:
            error = abs(b - a)
        
        r_anterior = r

        tabla.append({
            "iteracion": iteracion,
            "a": a,
            "b": b,
            "r": r,
            "f(r)": fr,
            "f(a)": fa,
            "f(b)": fb,
            "error": error,
            "metodo": "Illinois"
        })

        if abs(fr) < tolerancia or error < tolerancia:
            return r, fr, tabla, iteracion

        if fa * fr < 0:
            b = r
            fb = fr
        else:
            a = r
            fa = fr

    r = a - fa * (b - a) / (fb - fa)
    return r, f(r), tabla, max_iteraciones


# ---------------------------------------------------------
# Impresión de tabla - Método Estándar
# ---------------------------------------------------------
def imprimir_tabla_estandar(tabla):
    print("\nTABLA DE ITERACIONES - MÉTODO ESTÁNDAR")
    print("-" * 115)
    print(
        f"{'Iteración':>10} {'a':>15} {'b':>15} "
        f"{'r':>15} {'f(r)':>15} {'Error':>15}"
    )
    print("-" * 115)

    for fila in tabla:
        print(
            f"{fila['iteracion']:>10} "
            f"{fila['a']:>15.8f} "
            f"{fila['b']:>15.8f} "
            f"{fila['r']:>15.8f} "
            f"{fila['f(r)']:>15.8e} "
            f"{fila['error']:>15.8e}"
        )

    print("-" * 115)


# ---------------------------------------------------------
# Impresión de tabla - Método Illinois
# ---------------------------------------------------------
def imprimir_tabla_illinois(tabla):
    print("\nTABLA DE ITERACIONES - MÉTODO ILLINOIS (MODIFICADO)")
    print("-" * 140)
    print(
        f"{'Iter':>6} {'a':>12} {'b':>12} "
        f"{'r':>12} {'f(r)':>12} {'f(a)':>12} {'f(b)':>12} {'Error':>12}"
    )
    print("-" * 140)

    for fila in tabla:
        print(
            f"{fila['iteracion']:>6} "
            f"{fila['a']:>12.6f} "
            f"{fila['b']:>12.6f} "
            f"{fila['r']:>12.6f} "
            f"{fila['f(r)']:>12.6e} "
            f"{fila['f(a)']:>12.6e} "
            f"{fila['f(b)']:>12.6e} "
            f"{fila['error']:>12.6e}"
        )

    print("-" * 140)


# ---------------------------------------------------------
# Análisis del estancamiento
# ---------------------------------------------------------
def analizar_estancamiento(tabla, metodo):
    print(f"\nANÁLISIS DE ESTANCAMIENTO - {metodo}")
    print("-" * 80)
    
    contador_a_fijo = 0
    contador_b_fijo = 0
    iteraciones_a_fijo = []
    iteraciones_b_fijo = []
    
    for i in range(1, len(tabla)):
        if abs(tabla[i]['a'] - tabla[i-1]['a']) < 1e-15:
            contador_a_fijo += 1
            iteraciones_a_fijo.append(tabla[i]['iteracion'])
        if abs(tabla[i]['b'] - tabla[i-1]['b']) < 1e-15:
            contador_b_fijo += 1
            iteraciones_b_fijo.append(tabla[i]['iteracion'])
    
    print(f"Número de iteraciones donde 'a' permanece fijo: {contador_a_fijo}")
    print(f"Número de iteraciones donde 'b' permanece fijo: {contador_b_fijo}")
    
    if iteraciones_a_fijo:
        print(f"Iteraciones donde 'a' es fijo: {iteraciones_a_fijo[:10]}...")
    if iteraciones_b_fijo:
        print(f"Iteraciones donde 'b' es fijo: {iteraciones_b_fijo[:10]}...")
    
    if contador_a_fijo > 0 or contador_b_fijo > 0:
        porcentaje_estancamiento = (
            (contador_a_fijo + contador_b_fijo) / (2 * len(tabla)) * 100
        )
        print(f"\nEstancamiento total: {porcentaje_estancamiento:.1f}%")
        
        if porcentaje_estancamiento > 50:
            print("⚠️  ESTANCAMIENTO SEVERO DETECTADO")
        elif porcentaje_estancamiento > 20:
            print("⚠️  ESTANCAMIENTO MODERADO DETECTADO")
        else:
            print("⚠️  ESTANCAMIENTO LEVE DETECTADO")
    else:
        print("\n✓ No se detectó estancamiento significativo.")


# ---------------------------------------------------------
# Explicación del estancamiento
# ---------------------------------------------------------
def explicar_estancamiento():
    print("\n" + "="*80)
    print("¿POR QUÉ SE PRODUCE EL ESTANCAMIENTO?")
    print("="*80)
    print("""
ANÁLISIS MATEMÁTICO:

1. FÓRMULA DE FALSA POSICIÓN:
   r = a - f(a) × (b - a) / (f(b) - f(a))

2. PROBLEMA EN f(x) = x³ + x² - 1:
   
   - La derivada es: f'(x) = 3x² + 2x
   - En el intervalo [0, 2], la función es convexa
   - Esto causa que la recta secante siempre toque cerca del mismo extremo
   
3. COMPORTAMIENTO OBSERVADO:
   
   - f(0) = -1 (negativo)
   - f(2) = 11 (positivo)
   - La raíz real está cerca de x ≈ 0.7549
   
   - En la primera iteración:
     r ≈ 0.154 (cercano a 0)
   - Como f(0) < 0 y f(r) < 0, el siguiente intervalo es [0.154, 2]
   
   - El extremo derecho 'b=2' PERMANECE FIJO
   - La recta secante sigue teniendo pendiente similar
   - Convergencia lenta desde la izquierda

4. CAUSA RAÍZ:
   La función tiene una curvatura pronunciada (convexa), lo que causa que
   la interpolación lineal converja principalmente desde UN lado, dejando
   el otro extremo "estancado" durante muchas iteraciones.

5. IMPACTO:
   - Convergencia lenta
   - Muchas iteraciones innecesarias
   - Computacionalmente ineficiente
""")


# ---------------------------------------------------------
# Explicación de la solución (Illinois Method)
# ---------------------------------------------------------
def explicar_illinois():
    print("\n" + "="*80)
    print("SOLUCIÓN: ILLINOIS METHOD (FALSA POSICIÓN MODIFICADA)")
    print("="*80)
    print("""
¿CÓMO SOLUCIONA EL ILLINOIS METHOD EL ESTANCAMIENTO?

1. PRINCIPIO:
   Cuando un extremo permanece fijo en dos iteraciones consecutivas,
   se REDUCE su valor funcional por un factor de 0.5:
   
   - Si 'b' es fijo: fb = fb / 2
   - Si 'a' es fijo: fa = fa / 2

2. EFECTO GEOMÉTRICO:
   
   Método Estándar:       Método Illinois:
   
   f(b) = 11              f(b) = 5.5  (reducido)
   
   Esto "acerca" la recta secante al eje x, haciendo que:
   - El nuevo punto r se desplace más hacia el extremo fijo
   - La convergencia sea MÁS RÁPIDA

3. INTUICIÓN MATEMÁTICA:
   
   La fórmula de interpolación lineal es:
   r = a - f(a) × (b - a) / (f(b) - f(a))
   
   Al reducir f(b), el término (f(b) - f(a)) disminuye en valor absoluto,
   lo que amplifica el desplazamiento de r hacia b.

4. VENTAJAS:
   ✓ Reduce significativamente el número de iteraciones
   ✓ Converge más rápidamente que el método estándar
   ✓ Mantiene la robustez del método de bisección
   ✓ Simple de implementar

5. CASOS DE USO:
   - Funciones altamente cóncavas o convexas
   - Casos donde la curvatura varía mucho en el intervalo
   - Cuando se necesita convergencia rápida con seguridad
""")


# ---------------------------------------------------------
# Gráfica comparativa
# ---------------------------------------------------------
def graficar_comparacion(a, b, raiz_est, raiz_ill, tabla_est, tabla_ill):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfica 1: Función y raíces
    ax = axes[0, 0]
    margen = 0.5
    x = np.linspace(a - margen, b + margen, 1000)
    y = f(x)

    ax.plot(x, y, label=r"$f(x)=x^3+x^2-1$", color="blue", linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.axvspan(a, b, color="orange", alpha=0.2, label=f"Intervalo [{a}, {b}]")

    ax.scatter(raiz_est, f(raiz_est), color="red", s=100, zorder=5,
               label=f"Raíz (Estándar) = {raiz_est:.8f}")
    ax.scatter(raiz_ill, f(raiz_ill), color="green", s=100, marker='^', zorder=5,
               label=f"Raíz (Illinois) = {raiz_ill:.8f}")

    ax.set_title("Función y Raíces Encontradas", fontsize=12, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gráfica 2: Convergencia del error
    ax = axes[0, 1]
    iter_est = [fila['iteracion'] for fila in tabla_est]
    error_est = [fila['error'] for fila in tabla_est]
    iter_ill = [fila['iteracion'] for fila in tabla_ill]
    error_ill = [fila['error'] for fila in tabla_ill]

    ax.semilogy(iter_est, error_est, marker='o', linestyle='-', color='red',
                linewidth=2, label='Método Estándar')
    ax.semilogy(iter_ill, error_ill, marker='^', linestyle='-', color='green',
                linewidth=2, label='Método Illinois')

    ax.set_title("Convergencia: Error vs Iteración", fontsize=12, fontweight='bold')
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Error aproximado (escala logarítmica)")
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # Gráfica 3: Evolución del intervalo (Método Estándar)
    ax = axes[1, 0]
    iteraciones_est = list(range(1, len(tabla_est) + 1))
    b_values_est = [fila['b'] for fila in tabla_est]

    ax.plot(iteraciones_est, b_values_est, marker='o', color='red', linewidth=2,
            label='Extremo b (Estándar)')
    ax.axhline(raiz_est, color='red', linestyle='--', alpha=0.5,
               label=f'Raíz = {raiz_est:.6f}')
    ax.set_title("Evolución del Extremo b - Método Estándar", fontsize=12, fontweight='bold')
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Valor de b")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Gráfica 4: Evolución del intervalo (Método Illinois)
    ax = axes[1, 1]
    iteraciones_ill = list(range(1, len(tabla_ill) + 1))
    b_values_ill = [fila['b'] for fila in tabla_ill]

    ax.plot(iteraciones_ill, b_values_ill, marker='^', color='green', linewidth=2,
            label='Extremo b (Illinois)')
    ax.axhline(raiz_ill, color='green', linestyle='--', alpha=0.5,
               label=f'Raíz = {raiz_ill:.6f}')
    ax.set_title("Evolución del Extremo b - Método Illinois", fontsize=12, fontweight='bold')
    ax.set_xlabel("Número de iteración")
    ax.set_ylabel("Valor de b")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Comparación de resultados
# ---------------------------------------------------------
def comparar_resultados(raiz_est, iter_est, tabla_est, raiz_ill, iter_ill, tabla_ill):
    print("\n" + "="*80)
    print("COMPARACIÓN DE RESULTADOS")
    print("="*80)

    print(f"\n{'Métrica':<40} {'Estándar':>20} {'Illinois':>20}")
    print("-"*80)
    
    print(f"{'Raíz aproximada':<40} {raiz_est:>20.10f} {raiz_ill:>20.10f}")
    print(f"{'Diferencia de raíces':<40} {abs(raiz_est - raiz_ill):>20.10e} -")
    print(f"{'Iteraciones requeridas':<40} {iter_est:>20} {iter_ill:>20}")
    print(f"{'Reducción de iteraciones':<40} {'-':>20} {(1 - iter_ill/iter_est)*100:>19.1f}%")
    print(f"{'Error final':<40} {tabla_est[-1]['error']:>20.10e} {tabla_ill[-1]['error']:>20.10e}")

    # Análisis de estancamiento
    est_est = sum(1 for i in range(1, len(tabla_est)) if abs(tabla_est[i]['b'] - tabla_est[i-1]['b']) < 1e-15)
    est_ill = sum(1 for i in range(1, len(tabla_ill)) if abs(tabla_ill[i]['b'] - tabla_ill[i-1]['b']) < 1e-15)
    
    print(f"{'Iteraciones con 'b' fijo':<40} {est_est:>20} {est_ill:>20}")
    print(f"{'Reducción de estancamiento':<40} {'-':>20} {(1 - est_ill/est_est)*100 if est_est > 0 else 0:>19.1f}%")

    print("\n" + "="*80)
    print("CONCLUSIÓN")
    print("="*80)
    if iter_ill < iter_est:
        mejora = ((iter_est - iter_ill) / iter_est) * 100
        print(f"✓ El método Illinois es {mejora:.1f}% más eficiente que el método estándar")
        print(f"✓ Se requieren {iter_est - iter_ill} iteraciones menos")
    else:
        print("✓ Ambos métodos convergen en un número similar de iteraciones")


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------
if __name__ == "__main__":

    # Parámetros configurables
    a = 0
    b = 2
    tolerancia = 1e-5
    max_iteraciones = 100

    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO: FALSA POSICIÓN ESTÁNDAR vs ILLINOIS METHOD")
    print("="*80)
    print(f"Función: f(x) = x³ + x² - 1")
    print(f"Intervalo: [{a}, {b}]")
    print(f"Tolerancia: {tolerancia}")
    print(f"Máximo de iteraciones: {max_iteraciones}")

    print("\n" + "-"*80)
    print("VERIFICACIÓN DEL INTERVALO")
    print("-"*80)
    print(f"f({a}) = {f(a):.10f}")
    print(f"f({b}) = {f(b):.10f}")
    print(f"Cambio de signo: {'✓ SÍ' if f(a) * f(b) < 0 else '✗ NO'}")

    # Explicaciones
    explicar_estancamiento()
    explicar_illinois()

    # Método Estándar
    print("\n" + "="*80)
    print("EJECUCIÓN: MÉTODO ESTÁNDAR")
    print("="*80)
    raiz_est, valor_est, tabla_est, iter_est = falsa_posicion_estandar(
        a, b, tolerancia, max_iteraciones
    )
    
    print(f"\nRaíz aproximada: {raiz_est:.10f}")
    print(f"f(raíz): {valor_est:.10e}")
    print(f"Iteraciones: {iter_est}")
    imprimir_tabla_estandar(tabla_est)
    analizar_estancamiento(tabla_est, "MÉTODO ESTÁNDAR")

    # Método Illinois
    print("\n" + "="*80)
    print("EJECUCIÓN: MÉTODO ILLINOIS")
    print("="*80)
    raiz_ill, valor_ill, tabla_ill, iter_ill = falsa_posicion_illinois(
        a, b, tolerancia, max_iteraciones
    )
    
    print(f"\nRaíz aproximada: {raiz_ill:.10f}")
    print(f"f(raíz): {valor_ill:.10e}")
    print(f"Iteraciones: {iter_ill}")
    imprimir_tabla_illinois(tabla_ill)
    analizar_estancamiento(tabla_ill, "MÉTODO ILLINOIS")

    # Comparación
    comparar_resultados(raiz_est, iter_est, tabla_est, raiz_ill, iter_ill, tabla_ill)

    # Gráficas
    graficar_comparacion(a, b, raiz_est, raiz_ill, tabla_est, tabla_ill)
