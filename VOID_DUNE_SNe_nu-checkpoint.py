import numpy as np
import scipy as sp
import scipy.integrate
import scipy.interpolate

#def interp_1d_log_space(x, y):
 #   return sp.interpolate.interp1d(np.log10(x), y, kind='quadratic', bounds_error=False, fill_value='extrapolate')

def integrate_log_space(x, y, a, b):
    interp = sp.interpolate.interp1d(np.log10(x), y, kind='quadratic', fill_value='extrapolate')
    def integrand(log_x):
        return interp(log_x) * np.log(10) * 10**log_x
    return sp.integrate.quad(integrand, np.log10(a), np.log10(b), limit=500)

class get_data_from_Murase():
    
    def __init__(self, filename):
        block_data = []
        with open(filename, "r") as f:
            block = []
            for line in f:
                if line.strip():
                    block.append([float(x) for x in line.strip().split()])
                elif block:
                    block_data.append(np.array(block))
                    block = []
            if block:
                block_data.append(np.array(block))
                
        self.dN_dE_CR = np.array([[row[7] * row[13] * 1e9 for row in block] for block in block_data])
        self.energy_vals = np.array([row[0] / 1e9 for row in block_data[0]]) # convert eV -> GeV
        
        if "/IIn/" in filename or "II-PD1_0Rw1e15" in filename:
            self.times = [1e5 * 10**(0.1 * k) for k in range(len(self.dN_dE_CR))]
        else:
            self.times = [1e3 * 10**(0.1 * k) for k in range(len(self.dN_dE_CR))]
            
    def time_integral_dN_dE(self):
        time_integrated_dN_dE = np.zeros(384)
        dN_dE_tot = self.dN_dE_CR
            
   #     def energy_slice_interp(log_t, e):
    #        dN_dE = dN_dE_tot[e]
     #       interp = sp.interp(e, np.log10(self.times), dN_dE, fill_value="extrapolate", kind="cubic")
      #      return interp(log_t) * np.log(10) * 10**log_t

        for e in range(384):
            dN_dE = dN_dE_tot [:, e]
            integral = integrate_log_space(self.times, dN_dE, self.times[0], 100*24*3600)[0]
            time_integrated_dN_dE[e] = integral
        return time_integrated_dN_dE


class make_cs():
    
    def __init__(self):
        self.nu_mu_cs_cc_n = np.loadtxt("cs_files/nu_mu_H2_cc_n.txt", dtype=float)
        self.nu_mu_cs_cc_p = np.loadtxt("cs_files/nu_mu_H2_cc_p.txt", dtype=float)
        self.nu_mu_bar_cs_cc_n = np.loadtxt("cs_files/nu_mu_bar_H2_cc_n.txt", dtype=float)
        self.nu_mu_bar_cs_cc_p = np.loadtxt("cs_files/nu_mu_bar_H2_cc_p.txt", dtype=float)
        
    def get_nu_mu_cs_cc_n(self, energy):
        return np.interp(energy, self.nu_mu_cs_cc_n[:, 0], self.nu_mu_cs_cc_n[:, 1] * 1e-38)
    
    def get_nu_mu_cs_cc_p(self, energy):
        return np.interp(energy, self.nu_mu_cs_cc_p[:, 0], self.nu_mu_cs_cc_p[:, 1] * 1e-38)
    
    def get_nu_mu_bar_cs_cc_n(self, energy):
        return np.interp(energy, self.nu_mu_bar_cs_cc_n[:, 0], self.nu_mu_bar_cs_cc_n[:, 1] * 1e-38)
    
    def get_nu_mu_bar_cs_cc_p(self, energy):
        return np.interp(energy, self.nu_mu_bar_cs_cc_p[:, 0], self.nu_mu_bar_cs_cc_p[:, 1] * 1e-38)
    
    def get_nu_mu_cs_cc_avg(self, energy):
        return 0.5 * (self.get_nu_mu_cs_cc_n(energy) + self.get_nu_mu_cs_cc_p(energy))
    
    def get_nu_mu_bar_cs_cc_avg(self, energy):
        return 0.5 * (self.get_nu_mu_bar_cs_cc_n(energy) + self.get_nu_mu_bar_cs_cc_p(energy))

    
class detect_SNe_nu(get_data_from_Murase, make_cs):
    
    def __init__(self, filename, cross_section='cc'):
        get_data_from_Murase.__init__(self, filename)
        make_cs.__init__(self)
        self.cs = cross_section
        self.nu_mu_cs_interp = self.get_nu_mu_cs_cc_avg
        self.nu_mu_bar_cs_interp = self.get_nu_mu_bar_cs_cc_avg
        self._time_integrated_dN_dE = self.time_integral_dN_dE()
        
    def num_nucleons_DUNE(self):
        tons = 40000
        kg = tons * 1000
        g = kg * 1000
        mol = g / 55.845
        atoms = mol * 6.02e23
        N_det = atoms * 56
        return N_det
    
    def sphere_area(self):
        radius_kpc = 10
        radius_cm = radius_kpc * 3.086e21
        area = 4 * np.pi * radius_cm**2
        return area
        
    def get_energy_integrand_dN(self, energy):
        interp_time_integral = np.interp(np.log10(energy), np.log10(self.energy_vals), self._time_integrated_dN_dE)
        dN = interp_time_integral * self.get_nu_mu_cs_cc_avg(energy) * self.num_nucleons_DUNE() / self.sphere_area()
        return dN
    
    def get_energy_integrand_dN_bar(self, energy):
        interp_time_integral = np.interp(np.log10(energy), np.log10(self.energy_vals), self._time_integrated_dN_dE)
        dN_bar = interp_time_integral * self.get_nu_mu_bar_cs_cc_avg(energy) * self.num_nucleons_DUNE() / self.sphere_area()
        return dN_bar
    
    def get_starting_events(self, energy_range):
        x = np.logspace(np.log10(100), np.log10(1e6), 1000)
        y = self.get_energy_integrand_dN(x)
        y_bar = self.get_energy_integrand_dN_bar(x)
    
    
        nu_events = integrate_log_space(x, y, energy_range[0], energy_range[-1])[0] / 2
        nu_bar_events = integrate_log_space(x, y_bar, energy_range[0], energy_range[-1])[0] / 2
        return nu_events + nu_bar_events
        