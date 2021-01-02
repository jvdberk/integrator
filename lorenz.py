def lorenz(t, x, y, z):
    """
    x proportional to convective intensity
    y proportional to temperature difference between descending and ascending
      air flows
    z the difference in vertical temperature (linearised)

    sigma = nu/kappa
    r     = Ra/Ra_c        
    b     = 4/(1 + a*a) 

    sigma = Prandtl number
    nu    = kinematic viscosity
    kappa = thermal diffusivity
    Ra    = Rayleigh number
    Ra_c  = critical Rayleigh number
    b     = geometric factor
    a     = buoyancy

    sigma = 10
    b     = 8./3
    """
    sigma = 10
    b     = 8./3
    r     = 28.

    dx = sigma * (y - x)
    dy = -x*z + r*x - y
    dz =  x*y - b*z

    return dx, dy, dz
