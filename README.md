# COBAR_project
Implement courtship chasing while keeping a specific distance to the other fly using visual detection of the distance to the other fly

TODO: 
- finish sideway walking
- increase time of switch to crabe walking
- wings --> close and reopen after some time
- Plot to put in the report --> faire un notebook avec les différents comportements
- regler la camera quand la mouche sort du cadre 
- Ecrire readme
- documenter le code

## Prerequisites
the files contained in `data` must be added to the flygym-v1 repository for the provided code to run:
- `config.yaml` should replace the original config.yaml file contained in `flygym-v1/flygym`
- `neuromechfly_courtship_kinorder_ypr.yml` should be added in `flygym-v1/flygym/data/mjcf`

## Structure 
This repository contains different folders: 
- `data`: contains the two files to include in the flygym-v1 repository for the code to function 
- `scripts`: contains the definitions of the classes and functions implemented and used for the simulation
- `outputs`: contains the final videos 

The codes running the different simulations are in the notebooks: 
- `scenario.ipynb`: run the whole scenario of the courtship behavior
- `wings_vibrations.ipynb`: run the wing extension behavior 
- `crabe_wlaking.ipynb`: run the crabe walking 
- `chasing.ipynb`: run the chasing 

