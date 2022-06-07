# endmember_based_vegetation_characterization

This repository presents a method of using vegetation cycle characterizations to train an irrigation detection model. A full description of this methodology can be found in Chapter IV of the following [PhD dissertation](https://storage.googleapis.com/terry_phd_export/thesis/tconlon_phd_dissertation.pdf).

## Overview

The method of vegetation cycle characterization implemented here was first introduced in [Small (2012)](https://www.sciencedirect.com/science/article/pii/S0034425712002349), and involves first extracting temporal endmembers (tEMs) from the exterior of the point cloud representation of a region's vegetation phenologies. In this repository, [Climate Hazards Group InfraRed Precipitation with Station Data (CHIRPS)](https://www.chc.ucsb.edu/data/chirps) precipitation estimates are used in the endmember selection process: Two tEMs are selected as "in-phase" and "out-of-phase" with rainfall based on the CHIRPS timeseries.
These tEMs are then used to invert a linear mixture model, the pixelwise results of which constitute an "abundance map" and represent the contribution of each tEM to a vegetation signature. All spatiotemporal characterization is performed using MODIS 250m imagery, downloaded from [here]((https://lpdaacsvc.cr.usgs.gov/appeears/task/area)). 

The resulting endmember abundances at 250m resolution are used to train a classifier, with labels collected via visual inspection on Google Earth Pro. All endmembers, abundance maps, imagery, labels, and predictions are contained in the linked Google Cloud [bucket](https://console.cloud.google.com/storage/browser/terry_phd_export/projects/ethiopia/agu_irrigation_mapping/data_and_outputs).


## Repository Structure

```
ethiopia_irrigation_detection
├── chirps_processing.py
├── dataloader.py
├── environment.yml
├── main.py
├── model.py
├── params.yaml
├── plotting.py
├── spectral_unmixing.py
```

## Repository Description

Note: Each file described below also contains substantial line-by-line documentation. 

* `chirps_processing.py`: This script contains the functions necessary for processing CHIRPS rainfall estimates. These functions include those for loading data, clustering the rainfall timeseries, and extracting tEMs in-phase and out-of-phase with rainfall. 

* `dataloader.py`: This script contains functions necessary for reading in abundance maps, associating certain pixels in the maps with irrigated/non-irrigated labels, and preparing these labeled samples for model training. 

* `environment.yml`: This file specifies the Python packages required to run the code contained in this repository. Users can create the necessary Pyhton environment via `conda env create -f environment.yml`.

* `main.py`:  This is the main script for training the neural network (NN) based irrigation detection model. Run this script to execute the repository's functionality. 

* `model.py`: This file contains code that instantiates a simple deep NN to serve as the irrigation classification model. 

* `params.yaml`: This file contains user-specified parameters for model training.

* `plotting.py`: This script contains functions for plotting vegetation timeseries.

* `spectral_unmixing.py`:  This script contains code for applying a spectral unmixing model to determine the contribution of certain temproal endmembers to vegetation phenologies. See [Small (2012)](https://www.sciencedirect.com/science/article/pii/S0034425712002349) for a full description of the process of using endmember-based unmixing approaches for spatiotemporal vegetation characterization. 
