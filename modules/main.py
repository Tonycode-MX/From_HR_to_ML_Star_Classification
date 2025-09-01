from astroquery.gaia import Gaia
import numpy as np
import pandas as pd

Gaia.login()  # puedes hacerlo como guest si no te registras

job = Gaia.launch_job_async("""
SELECT TOP 30000
       ra, dec, parallax, parallax_error,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
       ruwe, phot_bp_rp_excess_factor
FROM gaiadr3.gaia_source
WHERE parallax_over_error > 10
  AND ruwe < 1.4
  AND parallax > 2
""")

results = job.get_results()
df = results.to_pandas()

df["BP_RP"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
d_pc = 1000.0 / df["parallax"]   # distancia en parsecs (parallax en mas)
df["M_G"] = df["phot_g_mean_mag"] + 5 - 5*np.log10(d_pc)
