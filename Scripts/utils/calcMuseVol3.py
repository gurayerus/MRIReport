#!/usr/bin/env python

### Import modules
import nibabel as nib
import numpy as np
import os, sys
import csv
import time

### Function to calculate derived ROI volumes for MUSE
def calcVolumes(roifile, mapcsv):
	
	### Check input mask
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

	### Calculate ROI count and volume
	ROIs, Counts = np.unique(roiimg, return_counts=True)
	Volumes = voxvol * Counts

	### Create an array indexed from 0 to max ROI index
	###   This array will speed up calculations for calculating derived ROIs
	###   Instead of adding ROIs in a loop, they are added at once using: 
	###          np.sum(all indexes for a derived ROI)  (see below)
	VolumesInd = np.zeros(ROIs.max()+1)
	VolumesInd[ROIs] = Volumes

	### Calculate derived volumes
	DerivedROIs = []
	DerivedVols = []
	ROIlist = []
	Vollist = []
	with open(mapcsv) as mapcsvfile:
		reader = csv.reader(mapcsvfile, delimiter=',')
		
		# Read each line in the csv map files
		for row in reader:			
			# Append the ROI number to the list
			DerivedROIs.append(row[0])
			roiInds = [int(x) for x in row[2:]]
			DerivedVols.append(np.sum(VolumesInd[roiInds]))

	### Decide what to output
	return DerivedROIs, DerivedVols
