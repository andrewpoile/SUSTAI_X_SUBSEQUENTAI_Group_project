# SUSTAI_X_SUBSEQUENTAI_Group_project

If using a windows machine, run in WSL virtual linux environment otherwise the depth_pro dependencies won't work. Use conda to activate the SUST.yml virtual environment and install dependencies.
Currently only depth pro is being imported locally.

## Files
- SUST.yml: list of dependencies and modules, pass to conda for activation.
- YOLOxDEPTHPRO.ipynb: contains the code used to run object detection, depth inference, and DBSCAN for group detections.
- YOLOxDEPTHPRO_1.ipynb: contains parallelised grouping code with depth-enabled interobject, metric distance estimation.
- YOLOxSpeedEstimation.ipynb: contains the code for running YOLO's speed estimation of the entire frame, which can be modified to extract the speed and the direction of detected objects.
- yolo_performance_testing.ipynb: contains the code used to investigate the depth pro speed discrepancies.
- performance.py: same as above but as a python script.
- projection_testing.ipynb: contains code to test the pinhole camera formula to estimate the size of boxes to then calculate inter-object distances. Heights of people seem to be within 20% of real heights. Depth estimation is tempered by passing the f35 equivalent of the camera used to film the videos used for testing, pending testing on the improvement to actual depth accuracy.

## Folders
- Homebrew-image: contains images used to test depth pro functionality.
- Homebrew-video: contains videos used to test object detection, depth and grouping algorithms.
- Speed_Results: contains results of speed estimation tests conducted to attempt to extract velocity from the yolo tracking module. Additionally, it includes folder called Mod_8. This folder includes the code for changing YOLO's Speed Estimation module. Once this change is made, the speed estimation will calculate the speed of detected objects in pixels per frame, as well as the direction of the detected objects, as an angle. For more information read the ReadMe file in folder Mod_8.
- runs: contains tracking results from YOLO and the DBSCAN grouping algorithm.
