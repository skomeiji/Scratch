import nuflux
import matplotlib.pyplot as plt
import numpy as np

flux = nuflux.makeFlux('H3a_SIBYLL23C')
nu_type=nuflux.NuMu

nu_energy = 1.5998587196060574
# colors = ['blue', 'green', 'red']
zenith = np.linspace(0,180,1000)
nu_cos_zenith = np.cos(np.radians(zenith))
nu_mu_flux = np.array([flux.getFlux(nu_type,nu_energy,c) for c in nu_cos_zenith])

plt.semilogy(nu_cos_zenith, nu_mu_flux, label=r'$\nu_\mu$')

# for E, color in zip(nu_energy, colors):
    # energy = E
    # nu_mu_flux = np.array([flux.getFlux(nu_type,energy,c) for c in nu_cos_zenith])
    # plt.semilogy(nu_cos_zenith, nu_mu_flux, label=fr'energy = {energy:.2f} GeV', color=color)

plt.xlabel('Zenith [radians]')
plt.ylabel(r'Flux [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
plt.title(r'$\nu_\mu$ Flux at 1 GeV')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("nu_mu_flux_plot_zenith_1GeV.png", dpi=300)