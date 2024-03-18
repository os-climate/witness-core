import numpy as np
import matplotlib.pyplot as plt

tp_a1 = 20.46
tp_a2 = 2.
tp_a3 = 6.081
tp_a4 = 6.4

def compute_damage(temp_atmo, tp_a1, tp_a2, tp_a3, tp_a4):
    dam = (temp_atmo / tp_a1) ** tp_a2 + (temp_atmo / tp_a3) ** tp_a4
    damage_frac_output = 1. - (1. / (1. + dam))
    return damage_frac_output

def plot():
    temp_atmo = np.arange(0., 10., 0.1)
    for tp_a3 in [2., 3., 4., 5., 6.]:
        damage_frac_output = compute_damage(temp_atmo, tp_a1, tp_a2, tp_a3, tp_a4)
        plt.plot(temp_atmo, damage_frac_output, label=f'{tp_a3}°C')

    plt.xlabel("Delta Temp [°C]")
    plt.ylabel("Damage fraction [-]")
    plt.title('Impact of tipping point value on damage fraction of GDPu')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot()