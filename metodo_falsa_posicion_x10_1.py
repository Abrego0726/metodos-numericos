import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Función asignada por la profesora
# f(x) = x^10 - 1
# ---------------------------------------------------------
def f(x):
    return x**10 - 1


# ---------------------------------------------------------
# Método de Falsa Posición estándar (Regula Falsi)
# ---------------------------------------------------------
def falsa_posicion_estandar(a, b, tolerancia=1e-5, max_iteraciones=100):
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, f(a), [], 0
    if fb == 0:
        return b, f(b), [], 0
    if fa * fb > 0:
        raise ValueError("El intervalo no es válido: f(a) y f(b) deben tener signos opuestos.")

    tabla = []
    r_anterior = None

    for iteracion in range(1, max_iteraciones + 1):
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
            "extremo_fijo": ""
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
# Método de Illinois (versión modificada de falsa posición)
# ---------------------------------------------------------
def falsa_posicion_illinois(a, b, tolerancia=1e-5, max_iteraciones=100):
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, f(a), [], 0
    if fb == 0:
        return b, f(b), [], 0
    if fa * fb > 0:
        raise ValueError("El intervalo no es válido: f(a) y f(b) deben tener signos opuestos.")

    tabla = []
    r_anterior = None

    for iteracion in range(1, max_iteraciones + 1):
        r = a - fa * (b - a) / (fb - fa)
        fr = f(r)

        if r_anterior is not None:
            error = abs(r - r_anterior)
        else:
            error = abs(b - a)

        # Detección del extremo fijo (estancamiento):
        # si el extremo no cambia entre iteraciones, se divide su valor funcional por 2
        if iteracion > 1:
            if abs(a - tabla[-1]["a"]) < 1e-15 and fa * fr > 0:
                fa = fa / 2.0
            if abs(b - tabla[-1]["b"]) < 1e-15 and fb * fr > 0:
                fb = fb / 2.0

        r_anterior = r

        tabla.append({
            "iteracion": iteracion,
            "a": a,
            "b": b,
            "r": r,
            "f(r)": fr,
            "error": error,
            "fa": fa,
            "fb": fb,
            "extremo_fijo": ""
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
# Presentación de tablas
# ---------------------------------------------------------
def imprimir_tabla(tabla, metodo):
    print(f"\nTABLA DE ITERACIONES - {metodo}")
    print("-" * 110)
    print(f"{'Iteración':>10} {'a':>15} {'b':>15} {'r':>15} {'f(r)':>15} {'Error':>15}")
    print("-" * 110)

    for fila in tabla:
        print(
            f"{fila['iteracion']:>10} "
            f"{fila['a']:>15.8f} "
            f"{fila['b']:>15.8f} "
            f"{fila['r']:>15.8f} "
            f"{fila['f(r)']:>15.8e} "
            f"{fila['error']:>15.8e}"
        )

    print("-" * 110)


# ---------------------------------------------------------
# Explicación del estancamiento
# ---------------------------------------------------------
def explicar_estancamiento():
    print("\n" + "=" * 80)
    print("¿POR QUÉ APARECE EL ESTANCAMIENTO?")
    print("=" * 80)
    print("""
Para f(x) = x^10 - 1 en [0, 2], se cumple:
    f(0) = -1
    f(2) = 1023
Existe cambio de signo, luego hay una raíz en el intervalo.

La raíz real es x = 1.

En la falsa posición, cuando el valor de f(x) en un extremo sigue teniendo
el mismo signo que f(r), ese extremo no cambia y se repite muchas veces.
Eso hace que la aproximación avance desde solo un lado, lo que se conoce como
estancamiento.

Esto ocurre porque la función crece muy rápido para x > 1 y la secante
se mantiene casi paralela al eje x en cierto tramo del intervalo.
""")


# ---------------------------------------------------------
# Gráfica de la función y la raíz
# ---------------------------------------------------------
def graficar_funcion(a, b, raiz, metodo):
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    y = f(x)

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, label=r"$f(x)=x^{10}-1$", color="blue", linewidth=2)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.axvspan(a, b, color="orange", alpha=0.2, label=f"Intervalo [{a}, {b}]")

    plt.scatter(raiz, f(raiz), color="red", s=80, zorder=5,
                label=f"Raíz aproximada = {raiz:.8f}")
    plt.annotate(f"({raiz:.6f}, {f(raiz):.2e})", (raiz, f(raiz)),
                 textcoords="offset points", xytext=(10, 10))

    plt.title(f"Gráfica de f(x) y raíz - {metodo}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------
if __name__ == "__main__":
    a = 0.0
    b = 2.0
    tolerancia = 1e-5
    max_iteraciones = 50

    print("MÉTODO DE FALSA POSICIÓN PARA f(x) = x^10 - 1")
    print("=" * 80)
    print(f"Intervalo: [{a}, {b}]")
    print(f"Tolerancia: {tolerancia}")
    print(f"Máximo de iteraciones: {max_iteraciones}")
    print(f"f({a}) = {f(a)}")
    print(f"f({b}) = {f(b)}")
    print("Existe cambio de signo, por lo tanto hay raíz en el intervalo.")

    explicar_estancamiento()

    # Método estándar
    raiz_std, valor_std, tabla_std, iter_std = falsa_posicion_estandar(a, b, tolerancia, max_iteraciones)
    print("\n" + "=" * 80)
    print("RESULTADO - FALSA POSICIÓN ESTÁNDAR")
    print("=" * 80)
    print(f"Raíz aproximada: {raiz_std:.10f}")
    print(f"f(raíz): {valor_std:.10e}")
    print(f"Iteraciones realizadas: {iter_std}")
    imprimir_tabla(tabla_std, "ESTÁNDAR")
    graficar_funcion(a, b, raiz_std, "ESTÁNDAR")

    # Método Illinois
    raiz_ill, valor_ill, tabla_ill, iter_ill = falsa_posicion_illinois(a, b, tolerancia, max_iteraciones)
    print("\n" + "=" * 80)
    print("RESULTADO - MÉTODO ILLINOIS")
    print("=" * 80)
    print(f"Raíz aproximada: {raiz_ill:.10f}")
    print(f"f(raíz): {valor_ill:.10e}")
    print(f"Iteraciones realizadas: {iter_ill}")
    imprimir_tabla(tabla_ill, "ILLINOIS")
    graficar_funcion(a, b, raiz_ill, "ILLINOIS")

    print("\n" + "=" * 80)
    print("COMPARACIÓN FINAL")
    print("=" * 80)
    print(f"Método estándar:        {iter_std} iteraciones")
    print(f"Método Illinois:        {iter_ill} iteraciones")
    if iter_ill < iter_std:
        print(f"Illinois converge más rápido: {iter_std - iter_ill} iteraciones menos.")
    elif iter_ill == iter_std:
        print("Ambos métodos convergen con el mismo número de iteraciones.")
    else:
        print("El método estándar fue más rápido en este caso.")
