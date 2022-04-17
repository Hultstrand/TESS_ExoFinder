r"""
Script to download the lightcurves for the TIC id of each TCE from MAST.
If a TIC has multiple lightcurves, it will download multiple. 
The script takes a csv file containing a list of TIC ids and returns a curl script that 
can be used to download a shell file of the lightcurves through bash

Example usage:
  python lightcurve_download_script.py \
    --tess_csv_file=tces.csv \
    --download_dir=${HOME}/astronet/tess

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pandas as pd
from astroquery.mast import Observations

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tess_csv_file",
    type=str,
    required=True,
    help="CSV file containing TESS targets to download. Must contain a "
    "'tic_id' column.")

parser.add_argument(
    "--download_dir",
    type=str,
    required=True,
    help="Directory into which the TESS data will be downloaded.")


def main(argv):
  del argv  # Unused.

  tcedata = pd.read_csv(FLAGS.tess_csv_file, comment='#', header=0)

  tic_list = tcedata['tic_id'].tolist() 
  num_tics = len(tic_list)
  
  tic_tess_observations = Observations.query_criteria(target_name=tic_list,obs_collection="TESS")

  tic_data_product = Observations.get_product_list(tic_tess_observations)

  filtered_tic_data_product = Observations.filter_products(tic_data_product, productType = "SCIENCE", productSubGroupDescription="LC")

  download_manifest = Observations.download_products(filtered_tic_data_product, download_dir=FLAGS.download_dir, curl_flag=True)
  
  print()
  print("\n{} TESS targets will be downloaded to {}".format(
      num_tics, FLAGS.download_dir))


  print("\nTo start the download, execute following shell file: " + download_manifest['Local Path'][0])

if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()
  main(argv=[sys.argv[0]] + unparsed)