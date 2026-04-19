from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
import numpy as np

# N_det calcuation
tons = 40000
kg = tons * 1000
g = kg * 1000
mol = g / 1.00845
atoms = mol * 6.02e23
N_det = atoms
N_det = atoms * 2 # 1 protons and 1 neutrons
print(f"number of atoms in det: {N_det}")

# distance calculation
radius_kpc = 10  # distance to SN
radius_cm = radius_kpc * 3.086e21
sphere_area = 4 * np.pi * radius_cm**2
print(f"surface area of sphere: {sphere_area}")

filenames = ["ModelTemplates/SNHEMM/IIn/FinalFluxs20.dat", "ModelTemplates/SNHEMM/II-P/FinalFluxs20.dat"]

# interpolate cross section data
with open("cs_files/nu_mu_H2_cc_n.txt", "r") as xsn, open("cs_files/nu_mu_H2_cc_p.txt", "r") as xsp, open("avg_xs.txt", "w") as avg:
    for line1, line2 in zip(xsn, xsp):
        if not line1.strip() or not line2.strip():
            continue
        # For line1
        parts1 = line1.strip().split()
        energy_n = float(parts1[0])
        xs_n = float(parts1[1]) if len(parts1) > 1 and parts1[1] != '' else 0.0

        # For line2
        parts2 = line2.strip().split()
        energy_p = float(parts2[0])
        xs_p = float(parts2[1]) if len(parts2) > 1 and parts2[1] != '' else 0.0
        
        if xs_n != 0 and xs_p != 0:
            xs = (xs_n + xs_p) / 2
            xs *= 1e-38
            avg.write(f"{energy_n} {xs}\n")
        elif xs_n != 0 and xs_p == 0:
            xs = xs_n * 1e-38
            avg.write(f"{energy_n} {xs_n}\n")
        elif xs_p != 0 and xs_n == 0:
            xs = xs_p * 1e-38
            avg.write(f"{energy_n} {xs_p}\n")
        else:
            avg.write(f"{energy_n}\n")

avg = np.loadtxt("avg_xs.txt")
energy_vals_xs = avg[:, 0]
xs_vals = avg[:, 1]
xs_interp = interp1d(np.log10(energy_vals_xs), xs_vals, kind='cubic', bounds_error=False, fill_value=0)

# integrate for each file
for file in filenames:
    block_data = []
    with open(file, "r") as f:
        block = []
        for line in f:
            if line.strip():
                block.append([float(x) for x in line.strip().split()])
            elif block:
                block_data.append(np.array(block))
                block = []
        if block:
            block_data.append(np.array(block))
        
        dN_dE_CR = np.array([[row[7]*row[13]* 1e9 for row in block] for block in block_data])
        energy_vals = np.array([row[0] / 1e9 for row in block_data[0]])
        
        if "IIn" in file:
            times = [1e5 * 10**(0.1 * k) for k in range(len(dN_dE_CR))]
        else:
            times = [1e3 * 10**(0.1 * k) for k in range(len(dN_dE_CR))]
            
        t_min = min(times)
        t_max = max(times) #100*24*3600
        
        e_min = 100
        e_max = 1e6
        
        print(f"min time: {t_min}, max time: {t_max}")
        print(len(dN_dE_CR))
        
        flux_interp = interp2d(np.log10(energy_vals), np.log10(times), dN_dE_CR, kind='cubic', fill_value=0) # convert to GeV
    
        # integrand functions
        def integrand(log_t, log_e):
            E = 10**log_e
            T = 10**log_t
            return (float(xs_interp(log_e)) *
                    float(flux_interp(log_e, log_t)) *
                    np.log(10)**2 * T * E)

        def inner_integral(log_e):
            result,error = quad(integrand, np.log10(t_min), np.log10(t_max), args=(log_e,))
            return result
    
        result, error = quad(inner_integral, np.log10(e_min), np.log10(e_max)) 
    
        result *= (N_det / sphere_area)
    
        print(f"{file}: {result}")

