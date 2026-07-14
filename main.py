# -*- coding: utf-8 -*-
"""

Ejecutar con:
    python main.py

O desde código externo:
    from raoi_simulator.farming import single_run, statistical_run

    report, objects_report, metrics = single_run(
        n_objects=10, n_plots=3, individuals=15,
        r_r=0.1, o_r=0.5, a_r=1.5,
        animation=True,
    )
"""

from raoi_simulator.farming import single_run as farm_single, statistical_run as farm_stat


def main() -> None:
    """Menú interactivo del simulador RAOI — solo Farming."""
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

        if choice == 1:
            farm_single()
            break
        elif choice == 2:
            try:
                replicas = int(input("Number of replicas: "))
            except ValueError:
                print("Invalid number.")
                continue
            farm_stat(replicas)
            break
        elif choice == 3:
            break
        else:
            print("Invalid option. Choose 1–3.")


if __name__ == "__main__":
    main()