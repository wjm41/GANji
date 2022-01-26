import os
import sys

repolocation = os.path.abspath(os.path.dirname(__file__))
home = os.path.expanduser("~")

#######################################################################################################
# Configuration definitions [ADD YOUR NEW CONFIGURATION HERE!]

# Automatically tries to determine config from repolocation
print(repolocation)
if repolocation == '/eos/home-d/drousso/SWAN_projects/Other/GANji':
    CONFIG = 'DAVID'
elif repolocation == '/home/wjm41/ml_physics/GANji':
    CONFIG = 'WILL-CSD3'
else:
    CONFIG = 'bruh'
print(CONFIG)

# Configuration definitions
if CONFIG == "DAVID":
    rds_dir = '/eos/home-d/drousso/SWAN_projects/Other'
elif CONFIG == "WILL-CSD3":
    rds_dir = '/rds-d2/user/wjm41/hpc-work/datasets'
else:
    print('ur a noob')
