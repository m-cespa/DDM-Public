from DDM import DDM

filepath = "... insert file path here ..."
filepath = "0.5micron_example.avi"

# specify pixel resolution (in μm per px for the lens in use)
pixel_size = 0.229
ddm_obj = DDM(filepath=filepath, pixel_size=pixel_size, particle_size=3, renormalise=True)

# number of points to sample between each power of 10 in time intervals
pointsPerDecade = 60
# recommended values: 30 for speed, 100 for accuracy
maxNCouples = 30

# generate list of indices log spaced
idts = ddm_obj.logSpaced(pointsPerDecade)

ddm_obj.calculate_isf(idts, maxNCouples, plot_heat_map=False)

ISF = ddm_obj.isf

ddm_obj.BrownianCorrelation(ISF, beta_guess=1.)
