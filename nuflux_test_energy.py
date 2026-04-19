import nuflux
import matplotlib.pyplot as plt
import numpy as np

flux = nuflux.makeFlux('H3a_SIBYLL23C')
nu_type=nuflux.NuE

nu_energies = np.logspace(-2, 10, 1000)
zenith = [0,45,90,180]
colors = ['blue', 'green', 'red', 'purple']
# nu_cos_zenith = np.cos(np.radians(zenith))
# nu_mu_flux = np.array([flux.getFlux(nu_type,E,nu_cos_zenith) for E in nu_energies])

fig, ax = plt.subplots(figsize=(12, 8))
plt.tight_layout(pad=2.0)

# plt.loglog(nu_energies, nu_mu_flux, label=r'$\nu_e$')

for angle, color in zip(zenith, colors):
    nu_cos_zenith = np.cos(np.radians(angle))
    nu_mu_flux = np.array([flux.getFlux(nu_type,E,nu_cos_zenith) for E in nu_energies])
    plt.loglog(nu_energies, nu_mu_flux, label=fr'$\cos(\theta)$ = {nu_cos_zenith:.2f}', color=color)

plt.xlabel('Energy [GeV]')
plt.ylabel(r'Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
plt.title(r'$\nu_e$ Flux')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()
plt.savefig("nu_e_flux_plot_energy°.png", dpi=300)