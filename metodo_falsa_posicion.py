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
# Método de Falsa Posición (Regula Falsi)
# ---------------------------------------------------------
def falsa_posicion(a, b, tolerancia=1e-5, max_iteraciones=100):
    """
    Método de Falsa Posición (Regula Falsi)
    
    Interpola linealmente entre los puntos (a, f(a)) y (b, f(b))
    para encontrar donde la línea cruza el eje x.
    
    Este método puede presentar estancamiento cuando uno de los
    extremos permanece fijo durante muchas iteraciones.
    """
    fa = f(a)
    fb = f(b)

    # Verificación del cambio de signo
    if fa == 0:
        return a, f(a), [], 0

    if fb == 0:
        return b, f(b), [], 0

    if fa * fb > 0:
        raise ValueError(
            "El intervalo no es válido: f(a) y f(b) deben tener signos opuestos."
        )

    tabla = []
    extremo_fijo = None

    for iteracion in range(1, max_iteraciones + 1):
        # Fórmula de la falsa posición (interpolación lineal)
        # r = a - f(a) * (b - a) / (f(b) - f(a))
        r = a - fa * (b - a) / (fb - fa)
        fr = f(r)

        # Error aproximado usando la diferencia de r entre iteraciones
        if iteracion == 1:
            error = abs(b - a)
            r_anterior = r
        else:
            error = abs(r - r_anterior)
            r_anterior = r

        # Detectar estancamiento
        if iteracion > 1:
            if a == tabla[-1]['a']:
                extremo_fijo = 'a'
            elif b == tabla[-1]['b']:
                extremo_fijo = 'b'

        tabla.append({
            "iteracion": iteracion,
            "a": a,
            "b": b,
            "r": r,
            "f(r)": fr,
            "error": error,
            "extremo_fijo": extremo_fijo
        })

        # Criterio de parada
        if abs(fr) < tolerancia or error < tolerancia:
            return r, fr, tabla, iteracion

        # Conservamos el subintervalo donde existe cambio de signo
        if fa * fr < 0:
            b = r
            fb = fr
        else:
            a = r
            fa = fr

    # Si se alcanza el máximo de iteraciones
    r = a - fa * (b - a) / (fb - fa)
    return r, f(r), tabla, max_iteraciones


# ---------------------------------------------------------
# Impresión de la tabla de iteraciones
# ---------------------------------------------------------
def imprimir_tabla(tabla):
    print("\nTABLA DE ITERACIONES")
    print("-" * 115)
    print(
        f"{'Iteración':>10} {'a':>15} {'b':>15} "
        f"{'r':>15} {'f(r)':>15} {'Error':>15} {'Extremo Fijo':>15}"
    )
    print("-" * 115)

    for fila in tabla:
        extremo_fijo_str = fila['extremo_fijo'] if fila['extremo_fijo'] else "---"
        print(
            f"{fila['iteracion']:>10} "
            f"{fila['a']:>15.8f} "
            f"{fila['b']:>15.8f} "
            f"{fila['r']:>15.8f} "
            f"{fila['f(r)']:>15.8e} "
            f"{fila['error']:>15.8e} "
            f"{extremo_fijo_str:>15}"
        )

    print("-" * 115)


# ---------------------------------------------------------
# Análisis del estancamiento
# ---------------------------------------------------------
def analizar_estancamiento(tabla):
    print("\nANÁLISIS DE ESTANCAMIENTO")
    print("-" * 60)
    
    contador_a_fijo = 0
    contador_b_fijo = 0
    
    for i in range(1, len(tabla)):
        if tabla[i]['a'] == tabla[i-1]['a']:
            contador_a_fijo += 1
        if tabla[i]['b'] == tabla[i-1]['b']:
            contador_b_fijo += 1
    
    print(f"Número de iteraciones donde 'a' permanece fijo: {contador_a_fijo}")
    print(f"Número de iteraciones donde 'b' permanece fijo: {contador_b_fijo}")
    
    if contador_a_fijo > 0 or contador_b_fijo > 0:
        print("\n⚠️  ESTANCAMIENTO DETECTADO")
        print("Este es un comportamiento típico del método de Falsa Posición")
        print("cuando uno de los extremos converge lentamente hacia la raíz.")
    else:
        print("\nNo se detectó estancamiento significativo.")


# ---------------------------------------------------------
# Gráfica de la función y la raíz encontrada
# ---------------------------------------------------------
def graficar_funcion(a, b, raiz, tabla):
    # Se amplía la gráfica para observar mejor la función
    margen = 0.5
    x = np.linspace(a - margen, b + margen, 1000)
    y = f(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfica 1: Función y raíz
    ax1.plot(x, y, label=r"$f(x)=x^3+x^2-1$", color="blue", linewidth=2)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axvline(0, color="gray", linewidth=1)

    # Intervalo utilizado
    ax1.axvspan(a, b, color="orange", alpha=0.2,
                label=f"Intervalo [{a:.4f}, {b:.4f}]")

    # Raíz aproximada
    ax1.scatter(
        raiz,
        f(raiz),
        color="red",
        s=100,
        zorder=5,
        label=f"Raíz aproximada x = {raiz:.8f}"
    )

    ax1.annotate(
        f"({raiz:.6f}, {f(raiz):.2e})",
        (raiz, f(raiz)),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
    )

    ax1.set_title("Método de Falsa Posición - Función y Raíz", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfica 2: Convergencia
    iteraciones = [fila['iteracion'] for fila in tabla]
    errores = [fila['error'] for fila in tabla]

    ax2.semilogy(iteraciones, errores, marker='o', linestyle='-', color='green', linewidth=2)
    ax2.set_title("Convergencia del Método", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Número de iteración")
    ax2.set_ylabel("Error aproximado (escala logarítmica)")
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------
if __name__ == "__main__":

    # Estos valores pueden cambiarse
    a = 0
    b = 2
    tolerancia = 1e-5
    max_iteraciones = 100

    print("MÉTODO DE FALSA POSICIÓN (REGULA FALSI)")
    print("=" * 80)
    print("Función: f(x) = x^3 + x^2 - 1")
    print(f"Intervalo seleccionado: [{a}, {b}]")
    print(f"Tolerancia: {tolerancia}")
    print(f"Máximo de iteraciones: {max_iteraciones}")

    print("\nVERIFICACIÓN DEL INTERVALO")
    print("-" * 80)
    print(f"f({a}) = {f(a):.10f}")
    print(f"f({b}) = {f(b):.10f}")

    if f(a) * f(b) < 0:
        print("✓ Existe un cambio de signo en el intervalo.")
        print("✓ Por el Teorema del Valor Intermedio, existe al menos una raíz.")
    elif f(a) == 0 or f(b) == 0:
        print("✓ Uno de los extremos del intervalo es una raíz.")
    else:
        print("✗ No existe cambio de signo en el intervalo.")
        raise SystemExit

    print("\nINFORMACIÓN ADICIONAL")
    print("-" * 80)
    print("Nota: Este método puede presentar estancamiento porque uno de los")
    print("extremos del intervalo puede permanecer fijo durante varias iteraciones.")
    print("Esto ocurre cuando la función tiene una curvatura muy pronunciada.")

    # Aplicación del método
    raiz, valor_funcion, tabla, cantidad_iteraciones = falsa_posicion(
        a,
        b,
        tolerancia,
        max_iteraciones
    )

    # Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    print(f"Raíz aproximada: {raiz:.10f}")
    print(f"f(raíz): {valor_funcion:.10e}")
    print(f"Cantidad de iteraciones realizadas: {cantidad_iteraciones}")

    imprimir_tabla(tabla)
    analizar_estancamiento(tabla)

    graficar_funcion(a, b, raiz, tabla)
