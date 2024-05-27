# Courtship chasing while keeping a specific distance using visual processing
BIOENG-456 - Controlling Behavior in Animals and Robots - 
Valentine Delevaux, Julie Le Tallec, Chléa Schiff.

## Project overview
The goal of the project is to implement courtship chasing while keeping a specific distance to the other fly using visual detection of the distance between flies. To this end, two fly classes have been defined, one for the female fly and one for the male fly. 

## Prerequisites
The files contained in `data` must be added to the flygym-v1 repository for the provided code to run:
- `config.yaml` should replace the original config.yaml file contained in `flygym-v1/flygym`
- `neuromechfly_courtship_kinorder_ypr.yml` should be added in `flygym-v1/flygym/data/mjcf`

To run the notebooks, use the flygym-v1 environment.

## Structure 
This repository contains two folders: 
- `data`: contains the two files to include in the flygym-v1 repository for the code to function 
- `scripts`: contains the definitions of the classes and functions implemented and used for the simulation

The codes used to run the different simulations are in the notebooks: 
- `1_scenario.ipynb`: run the whole scenario of the courtship behavior
- `2_chasing.ipynb`: run the chasing part
- `3_crab_walk_and_wings_extension.ipynb`: run the crab walking part along with wing extension behavior 
- `4_robustness_visual_processing`: studies the robustness of visual processing with colored terrain and flies

## Report and videos
All methods and results are described in the [Report](BIOENG_456-Report.pdf)
All videos produced using this code is on [the drive](https://drive.google.com/drive/u/1/folders/1RxnbTyR-75OWsPtSnCLu-PBjjf7ijszE).


