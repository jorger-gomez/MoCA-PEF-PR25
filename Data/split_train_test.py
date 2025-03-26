import argparse
import os
import shutil
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script para dividir subcarpetas completas en conjuntos de train y test.")
    parser.add_argument('--ruta_entrada', '-e', type=str, required=True, help="Ruta donde están las subcarpetas a dividir.")
    parser.add_argument('--ruta_salida', '-s', type=str, required=True, help="Ruta donde se creará la división entre train y test.")
    parser.add_argument('--train_size', '-t', type=float, required=True, help="Proporción de subcarpetas para entrenamiento (ej. 0.8).")
    parser.add_argument('--seed', type=int, default=None, help="Seed para reproducibilidad.")
    return parser.parse_args()

def copiar_carpeta(origen, destino):
    if os.path.exists(destino):
        shutil.rmtree(destino)
    shutil.copytree(origen, destino)

def separar_subcarpetas(ruta_entrada, ruta_salida, train_size, seed=None):
    if seed is not None:
        random.seed(seed)

    # Crear carpetas destino
    ruta_train = os.path.join(ruta_salida, "train")
    ruta_test = os.path.join(ruta_salida, "test")
    os.makedirs(ruta_train, exist_ok=True)
    os.makedirs(ruta_test, exist_ok=True)

    subcarpetas = [f for f in os.listdir(ruta_entrada) if os.path.isdir(os.path.join(ruta_entrada, f))]
    random.shuffle(subcarpetas)

    n_train = int(len(subcarpetas) * train_size)
    train_set = subcarpetas[:n_train]
    test_set = subcarpetas[n_train:]

    for sub in train_set:
        origen = os.path.join(ruta_entrada, sub)
        destino = os.path.join(ruta_train, sub)
        copiar_carpeta(origen, destino)

    for sub in test_set:
        origen = os.path.join(ruta_entrada, sub)
        destino = os.path.join(ruta_test, sub)
        copiar_carpeta(origen, destino)

    # Guardar conteo
    with open(os.path.join(ruta_salida, "conteo_subcarpetas.txt"), "w") as f:
        f.write(f"Total subcarpetas: {len(subcarpetas)}\n")
        f.write(f"Train: {len(train_set)} subcarpetas\n")
        f.write(f"Test: {len(test_set)} subcarpetas\n")
        f.write(f"Seed utilizada: {seed}\n")

    print("Separación completada con éxito.")

def main():
    args = parse_arguments()
    print(f"Dividiendo subcarpetas en {args.ruta_entrada} hacia {args.ruta_salida}...")
    separar_subcarpetas(args.ruta_entrada, args.ruta_salida, args.train_size, args.seed)
    print("¡Listo!")

if __name__ == "__main__":
    main()