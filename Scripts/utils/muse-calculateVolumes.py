#!/usr/bin/env python

### Import modules
import nibabel as nib
import numpy as np
import os, sys

mapcsv = '/home/guray/Documents/CBICA/Projects/DeepMRSeg/Kapaana/Modules/MRIReport/Dict/MUSE_DerivedROIs_Mappings.csv'

def calcVolume(roi):
	if not maskfile:
		return voxvol * np.count_nonzero( img[ img==roi ] )
	else:
		return voxvol * np.count_nonzero( img[ mask==roi ] )

def calcVolumes(roifile)
	### Sanity check on the arguments
	if not roifile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0) 

	### Read the input image
	roinii = nib.load(os.path.join(roifile))
	roiimg = roinii.get_data()
	roihdr = roinii.get_header()

	### Get voxel dimensions
	voxdims = roihdr.structarr["pixdim"]
	voxvol = float(voxdims[1]*voxdims[2]*voxdims[3])

	### ROIs to calculate volume for
	ROIs = np.unique(roiimg[np.nonzero(roiimg)])

	### Calculate volumes
	Volumes = []
	Volumes = np.array( list(map( calculateVolume, (ROIs) )) )

	DerivedROIs = []
	DerivedVols = []

	### Calculate derived volumes
	import csv

	with open(mapcsv) as mapcsvfile:
		reader = csv.reader(mapcsvfile, delimiter=',')
		
		# Read each line in the csv map files
		for row in reader:			
			# Append the ROI number to the list
			DerivedROIs.append(row[0])

			vol = 0
			for roi in range( 2,len(row) ):
				# Check if the roi exists in the list. If it does, calculate, else just say 0
				if ROIs[ ROIs == int(row[ roi ]) ]:
					vol += Volumes[ ROIs == int(row[ roi ]) ][0]

			DerivedVols.append(vol)

	### Decide what to output
	ROIlist.extend(DerivedROIs)
	Vollist.extend(DerivedVols)