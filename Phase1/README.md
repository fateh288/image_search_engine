
Folder and File Descriptions
1. Code- Contains all the code
- comments_main.py - ignore this file - contains debugging code
- constants.py - constants used in code
- data_loader_utils.py - utility functions to load images from folder into python data structures
- database_utils.py - utility functions for inserting, removing, selecting data from shelve database
- display_utils.py - utility functions to visualize images and/or results
- feature_extractor - classes for extracting features (color moment, HOG, ELBP)
- main.py - Entry point of project from where other functions are called.
- similarity_measures.py - utility functions for different similarity measures such as chi-square, p-norm etc.
- similarity_utils.py - utility functions for computing similarity and ranking of images.
- tasks.py - utility functions that correspond to tasks as defined in phase1. Most functions in other classes/files are called from this file.
- utils.py - general utility functions for arrays, dicts etc.
2. Datasets - Contains folders of images for which image recognition is required
3. Outputs - Human readable output. txt files for features and .png for similarity comparison images for queries.
4. Temp - Contains persistent files created by shelve database such as .bak,.dat,.dir files

System requirements/installation and execution instructions
1. Install Windows  
2. Setup python with Pycharm and Anaconda
3. Install python packages - numpy, scipy, matplotlib, sklearn, shelve, tqdm. 
4. Copy the ’Phase1’ folder which contains various folders including ’Code’, ’Datasets’etc.
5. In the ’Datasets’ directory, paste the folders ’set1’,’set2’ and ’set3’
6. Initialize ’foldername’ list in main.py to [’set1’,’set2’,’set3’] i.e. the folders that contain the images and we want extract features of.
7. Initialize ’queries’ list in main.py to [...  (’set2’, ’image-10’, ’color’, 4) ...]  i.e.  in formof tuple (foldername, queryimagefilename, featuretype, k)
8. Initialize write=False in main.py if features have already been extracted and stored in database and only requirement is to fetch ’k’ closest images for a given query.
9. Run the main.py file from ’Phase1’ as root folder using ’python Code/main.py’ command