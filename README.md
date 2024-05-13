# COBAR_project
Implement courtship chasing while keeping a specific distance to the other fly using visual detection of the distance to the other fly

TODO: 
- Implement sideway walking --> change dof, biases etc
- Implement wings UNILATERAL extension and vibration
- Plot of target vs chasing fly behavior (inspire from papers)
- Start wings vibration/crabe walking when the fly stops for target fly stops for some time
- stop vibrating after a specific time --> add a counter in attribute and reset the timesteps counter

## Prerequisites
the files contained in `data` must be added to the flygym-v1 repository for the provided code to run:
- `config.yaml` should replace the original config.yaml file contained in `flygym-v1/flygym`
- `neuromechfly_courtship_kinorder_ypr.yml` should be added in `flygym-v1/flygym/data/mjcf`
