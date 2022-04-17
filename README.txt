

Required Packages
Tensorflow
Pandas
Numpy
Astropy
Astroquery

TESS ExoFinder is designed to search through TESS lightcurves for planet signals. It will determine whether threshold crossing events (TCEs) are true planet candidates or false positives.
A TCE is a light curve that shows a characteristic dip in its brightness/flux which can correspond to the crossing of a planet across the telescope's view of the star. 
However, a TCE may be either a true planet or another explanation such as instrumental noise or eclipsing binaries. Therefore, TCEs must be analyzed to determine the veracity.
TESS ExoFinder was trained on light curves downloaded from the MAST database based on the TCEs and dispositions (labels) found in ExoFop database.

The following Python calls may be used to create the catalog that will be used to download light curves, generate TFREcords for training, and train the network.
Before creating an empty catalog, TIC IDs of the interested light curves must be gathered and placed in a .txt file. 
TCEs may be obtained via a sector bulk download from: https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html
They may also be optained from the ExoFop TESS database: https://exofop.ipac.caltech.edu/tess/
The dispositions of the TICs may be obtained from the ExoFop TESS database, manual assignment, or other databases.

The tce-list.txt tcesdatafile.txt, tces.csv, and TFRecords used in the original training of the model are included in the files.

The network accepts a local and global view of the light curve in two separate convolutional branches. They are then combined in a final fully connected block until a sigmoid output layers delivers a final prediction of a light curve being a planet candidate or not.

#Current path to TESS ExoFinder
    C:\Users\A_J_F\Documents\TESS_ExoFinder\data

#Create empty catalog
    #Input is a text file containing the list of TCEs
    #Output is empty_catalog.csv

    python create_empty_catalog.py --base_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data --input tce-list.txt --save_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data

#Create filled catalog
    #Inputs are the empty catalog and the tcestatfile from ExoFop/MAST
    #Ensure that the empty catalog and tcestatfile TCEs are in the same order in the csv files

    python create_catalog.py --input empty_catalog.csv --num_worker_processes=1 --tcestatfile tcestatfile.csv --base_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data --out_name=tces.csv

#Download lightcurves
    #Input the created catalog
    #Output a .sh file that can be ran to download lightcurve .fits files from MAST

    python lightcurve_download_script.py --tess_csv_file=tces.csv --download_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data\tess

#Build TFRecords for training
    #Input the created catalog and downloaded lightcurves
    #Output 8 training files, 1 validation file, and 1 test file
    
    python create_input_records.py --input_tce_csv_file=C:\Users\A_J_F\Documents\TESS_ExoFinder\data\tces.csv --tess_data_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data\tess --output_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data\tfrecord_double --num_worker_processes=1

#Train TESS ExoFinder
   #Input the TFRecords created from the TCE catalog
   #The batch size, epochs, learning rate, and prediction threshold may be set. The prediction threshold refers to the value above which the network will predict a planet versus a false positive in the final metrics
   #Output the accuracy, recall, precision, and F1 score of the network as recorded on the test data
   python TESS_ExoFinder.py --tfrecord_dir=C:\Users\A_J_F\Documents\TESS_ExoFinder\data\tfrecord_double --batch_size=32 --number_of_epochs=400 --learning_rate=0.000001 --prediction_threshold=0.7
