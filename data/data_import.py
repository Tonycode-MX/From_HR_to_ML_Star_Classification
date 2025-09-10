from astroquery.gaia import Gaia

def gaia_data_import(user: str = "", password: str = ""):
    """
    Import data from the Gaia archive using astroquery.

    If no username and password are provided, the function will automatically 
    connect as a *guest* user. Guest sessions allow easy access without 
    registration, but have important limitations:

        - Maximum of 30,000 rows can be retrieved per query.
        - Some advanced functionalities may be restricted compared to registered accounts.
    
    Note: All parameters and imported columns can be adjusted in data/data_importing.py

    Parameters
    ----------
    user : str, optional
        Gaia archive username. Leave empty to connect as guest.
    password : str, optional
        Gaia archive password. Leave empty to connect as guest.

    Returns
    -------
    None
        The function executes the query and returns the results object converted to a Pandas DataFrame.
    """

    Gaia.login(user = user, password= password)

    job = Gaia.launch_job_async("""
    -- GIANTS
    SELECT TOP 15
    ra, dec, parallax, parallax_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    ruwe, phot_bp_rp_excess_factor, phot_g_mean_flux,
    astrometric_excess_noise, bp_rp, bp_g, g_rp,random_index 
    FROM gaiadr3.gaia_source
    WHERE parallax_over_error > 10
    AND ruwe < 1.4
    AND (phot_bp_mean_mag - phot_rp_mean_mag) >= 0.8
    AND (phot_g_mean_mag - 10 + 5*LOG10(parallax)) <= 2.5
    ORDER BY random_index

    UNION ALL

    -- WDs
    SELECT TOP 15
    ra, dec, parallax, parallax_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    ruwe, phot_bp_rp_excess_factor, phot_g_mean_flux,
    astrometric_excess_noise, bp_rp, bp_g, g_rp, random_index
    FROM gaiadr3.gaia_source
    WHERE parallax_over_error > 10
    AND ruwe < 1.4
    AND (phot_bp_mean_mag - phot_rp_mean_mag) BETWEEN -0.5 AND 1.8
    AND (phot_g_mean_mag - 10 + 5*LOG10(parallax)) >= 10
    ORDER BY random_index

    UNION ALL

    -- MS
    SELECT TOP 15
    ra, dec, parallax, parallax_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    ruwe, phot_bp_rp_excess_factor, phot_g_mean_flux,
    astrometric_excess_noise, bp_rp, bp_g, g_rp, random_index
    FROM gaiadr3.gaia_source
    WHERE parallax_over_error > 10
    AND ruwe < 1.4
    AND NOT (
        ((phot_bp_mean_mag - phot_rp_mean_mag) >= 0.8
        AND (phot_g_mean_mag - 10 + 5*LOG10(parallax)) <= 2.5)
        OR
        ((phot_bp_mean_mag - phot_rp_mean_mag) BETWEEN -0.5 AND 1.8
        AND (phot_g_mean_mag - 10 + 5*LOG10(parallax)) >= 10)
    )
    ORDER BY random_index

    """)

    results = job.get_results()
    df = results.to_pandas()
    
    # Process results as needed
    print("\n==========================================")
    print("=====  Gaia data import successful!  =====")
    print("==========================================\n")
    print("Imported columns:", list(df.columns))
    print("\nNumber of rows imported:", len(df))
    print("\nNote: All parameters and imported columns can be adjusted in data/data_importing.py")


    return df