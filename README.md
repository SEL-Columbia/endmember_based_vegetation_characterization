# endmember_based_vegetation_characterization

This repository presents a method of using vegetation cycle characterizations to train an irrigation detection model. A full description of this methodology can be found in Chapter IV of the following [PhD dissertation](https://storage.googleapis.com/terry_phd_export/thesis/tconlon_phd_dissertation.pdf).

## Overview

The method of vegetation cycle characterization implemented here was first introduced in [Small (2012)](https://www.sciencedirect.com/science/article/pii/S0034425712002349), and involves first extracting temporal endmembers (tEMs) from the exterior of the point cloud representation of a region's vegetation phenologies. In this repository, [Climate Hazards Group InfraRed Precipitation with Station Data (CHIRPS)](https://www.chc.ucsb.edu/data/chirps) precipitation estimates are used in the endmember selection process: Two tEMs are selected as "in-phase" and "out-of-phase" with rainfall based on the CHIRPS timeseries.
These tEMs are then used to invert a linear mixture model, the pixelwise results of which constitute an "abundance map" and represent the contribution of each tEM to a vegetation signature. All spatiotemporal characterization is performed using MODIS 250m imagery, downloaded from [here]((https://lpdaacsvc.cr.usgs.gov/appeears/task/area)). 

The resulting endmember abundances at 250m resolution are used to train a classifier, with labels collected via visual inspection on Google Earth Pro. All endmembers, abundance maps, imagery, labels, and predictions are contained in the linked Google Cloud [bucket](https://console.cloud.google.com/storage/browser/terry_phd_export/projects/ethiopia/agu_irrigation_mapping/data_and_outputs).


## Repository Structure

## Repository Description



