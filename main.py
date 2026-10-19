"""Menú interactivo del simulador RAOI de farming.

Ejecutar con:
    python main.py

Desde código, ver raoi_simulator.farming: run(), single_run() y
statistical_run().
"""

import logging

from raoi_simulator.farming import single_run as farm_single, statistical_run as farm_stat

OPTION_SINGLE, OPTION_STATS, OPTION_EXIT = 1, 2, 3


def _run_option(choice: int) -> bool:
    """Ejecuta la opción elegida del menú.

    Args:
        choice: Opción del menú.

    Returns:
        True si el menú debe terminar; False si hay que volver a preguntar.
    """
    if choice == OPTION_SINGLE:
        farm_single()
        return True
    if choice == OPTION_STATS:
        try:
            replicas = int(input("Number of replicas: "))
        except ValueError:
            print("Invalid number.")
            return False
        if replicas < 1:
            print("Enter at least 1 replica.")
            return False
        farm_stat(replicas)
        return True
    if choice == OPTION_EXIT:
        return True
    print("Invalid option. Choose 1–3.")
    return False


def main() -> None:
    """Muestra el menú y ejecuta una simulación individual o una tanda de réplicas."""
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    menu = """
    ╔══════════════════════════════════════════════════╗
    ║       RAOI Swarm Simulator — Farming             ║
    ╠══════════════════════════════════════════════════╣
    ║    1. Single simulation                          ║
    ║    2. Statistical run (multiple replicas)        ║
    ║    3. Exit                                       ║
    ╚══════════════════════════════════════════════════╝
    """
    print(menu)

    while True:
        try:
            choice = int(input("Option: "))
        except ValueError:
            print("Please enter a number.")
            continue

        try:
            if _run_option(choice):
                break
        except ValueError as error:
            print(f"Invalid parameters: {error}")
            break


if __name__ == "__main__":
    main()
