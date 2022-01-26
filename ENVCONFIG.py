import os
import sys

repolocation=os.path.abspath(os.path.dirname(__file__))
home=os.path.expanduser("~")

#######################################################################################################
### Configuration definitions [ADD YOUR NEW CONFIGURATION HERE!]

### Automatically tries to determine config from repolocation
print(repolocation)
if repolocation=='/eos/home-d/drousso/SWAN_projects/Other/GANji':
    CONFIG='DAVID'
else:
    CONFIG='WILL'
print(CONFIG)

### Configuration definitions
if CONFIG=="DAVID":
    rds_dir = '/eos/home-d/drousso/SWAN_projects/Other'
elif CONFIG=="WILL":
    rds_dir = '/rds-d2/user/wjm41/hpc-work/datasets'