"""
Call Function
"""


from sbat.sbat import main,Model


#%% generate the object
#sbat=Model()


sbat=main(output=True)

""" 
def call_main():
    sbat=main(output=True)
import cProfile,pstats
profiler = cProfile.Profile()
profiler.enable()
call_main()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.strip_dirs()
stats.print_stats()
"""


#get baseflow
#sbat.get_baseflow()


#sbat.get_water_balance()




