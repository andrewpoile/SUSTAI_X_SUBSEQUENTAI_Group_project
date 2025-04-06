# SUSTAI_X_SUBSEQUENTAI_Group_project

If using a windows machine, run in WSL virtual linux environment otherwise the depth_pro dependencies won't work.

## Files
- YOLOxDEPTHPRO.ipynb: contains the code used to run object detection, depth inference, and DBSCAN for group detections.
- yolo_performance_testing.ipynb: contains the code used to investigate the depth pro speed discrepancies.
- performance.py: same as above but as a python script.
- projection_testing.ipynb: contains code to test the pinhole camera formula to estimate the size of boxes to then calculate inter-object distances.

## Folders
- Homebrew-image: contains images used to test depth pro functionality.
- Homebrew-video: contains videos used to test object detection, depth and grouping algorithms.
- Speed_Results: contains results of speed estimation tests conducted to attempt to extract velocity from the yolo tracking module.