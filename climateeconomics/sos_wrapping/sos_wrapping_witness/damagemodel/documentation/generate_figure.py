'''
Copyright 2024 Capgemini
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
import matplotlib.pyplot as plt
import numpy as np


def generate_fig_tipping_point_model(tp_a3):
    tp_a1 = 20.46
    tp_a2 = 2

    tp_a4 = 6.754

    def damage_func(temp):
        dam = (temp / tp_a1)**tp_a2 + (temp / tp_a3)**tp_a4
        return (1 - (1 / (1 + dam))) * 100

    N = 200
    temp = np.linspace(0, 6, N)

    damges = damage_func(temp)

    plt.figure(figsize=(10, 4))

    bullet_points = np.arange(0.5, 5.5, 0.5)

    plt.title(f'Tipping point damage model (Weitzmann, 2009) - tipping point {tp_a3}°C')
    plt.xlabel("Temperature increase (°C)")
    plt.ylabel("Impact on GDP (%)")
    plt.plot(temp, damges, zorder=-1)
    plt.grid()
    plt.axvline(tp_a3, alpha=0.4, c='r', label='Tipping point')

    for i, bp in enumerate(bullet_points):
        dm_bp = damage_func(bp)
        plt.scatter([bp], [dm_bp], label=f'{bp:.1f} - {dm_bp:.1f}%', alpha=(i + 1) / (len(bullet_points) + 1), c='r', zorder=4)
    plt.legend()
    from os import path

    curdir = path.dirname(__file__)
    fig_filename = path.join(curdir, f"tipping_point_damage_model{str(tp_a3).replace('.', '')}.png")
    plt.savefig(fig_filename)


generate_fig_tipping_point_model(3.5)
generate_fig_tipping_point_model(6.081)
