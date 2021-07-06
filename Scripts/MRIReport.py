import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv as _csv

import plotly as py
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats

from pygam import ExpectileGAM
from pygam.datasets import mcycle
from pygam import LinearGAM

#import plotly as py
#import plotly.plotly as py
#import plotly.tools as plotly_tools
#import plotly.graph_objs as go

from IPython.display import HTML


### Constants (TODO: temporary)
MUSE_ROI_Mapping = '/home/guray/Documents/CBICA/Projects/DeepMRSeg/Kapaana/Modules/MRIReport/Dict/MUSE_DerivedROIs_Mappings.csv'
MUSE_Ref_Values = '/home/guray/Documents/CBICA/Projects/DeepMRSeg/Kapaana/Modules/MRIReport/Dict/RefStudy_MRVals.csv'
SEL_ROI = ['MUSE_Volume_701', 'MUSE_Volume_601', 'MUSE_Volume_604', 'MUSE_Volume_509', 'MUSE_Volume_48']
SEL_ROI_Rename = ['MUSE_TotalBrain', 'MUSE_GM', 'MUSE_WM', 'MUSE_VN', 'MUSE_HippoL']

################################################ FUNCTIONS ################################################

############## HELP ##############


#DEF ARGPARSER
def read_flags():
	"""Parses args and returns the flags and parser."""
	### Import modules
	import argparse as _argparse

	parser = _argparse.ArgumentParser( formatter_class=_argparse.ArgumentDefaultsHelpFormatter )

	inputArgs = parser.add_argument_group('INPUT ARGS')

	inputArgs.add_argument( "--info", default=None, type=str, help="Subject info file")
	inputArgs.add_argument( "--bmask", default=None, type=str, help="brain mask image")
	inputArgs.add_argument( "--icv", default=None, type=str, help="ICV mask image")
	inputArgs.add_argument( "--roi", default=None, type=str, help="ROI segmentation image")
	
	### Read args
	flags = parser.parse_args()

	### Return flags and parser
	return flags, parser

### Function to calculate mask volume
def calcMaskVolume(maskfile):
	
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0) 

	### Read the input image
	roinii = nib.load(maskfile)
	roiimg = roinii.get_fdata()
	roihdr = roinii.header

	### Get voxel dimensions
	voxdims = roihdr.structarr["pixdim"]
	voxvol = float(voxdims[1]*voxdims[2]*voxdims[3])

	### Calculate mask volume
	maskVol = voxvol * np.sum(roiimg.flatten()>0)
	
	return maskVol

### Function to calculate derived ROI volumes for MUSE
def calcRoiVolumes(maskfile, mapcsv):
	
	### Check input mask
	if not maskfile:
		print("ERROR: Input file not provided!!!")
		sys.exit(0) 

	print(maskfile)

	### Read the input image
	roinii = nib.load(maskfile)
	roiimg = roinii.get_fdata()
	roihdr = roinii.header

	### Get voxel dimensions
	voxdims = roihdr.structarr["pixdim"]
	voxvol = float(voxdims[1]*voxdims[2]*voxdims[3])

	### Calculate ROI count and volume
	ROIs, Counts = np.unique(roiimg, return_counts=True)
	ROIs = ROIs.astype(int)
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
		reader = _csv.reader(mapcsvfile, delimiter=',')
		
		# Read each line in the csv map files
		for row in reader:			
			# Append the ROI number to the list
			DerivedROIs.append(row[0])
			roiInds = [int(x) for x in row[2:]]
			DerivedVols.append(np.sum(VolumesInd[roiInds]))

	### Decide what to output
	return dict(zip(DerivedROIs, DerivedVols))

def plotWithRef(dfRef, dfSub, selVar, fname):

	X = dfRef.Age.values.reshape([-1,1])
	y = dfRef[selVar].values.reshape([-1,1])

	##############################################################
	## Fit expectiles
	# fit the mean model first by CV
	gam50 = ExpectileGAM(expectile=0.5).gridsearch(X, y)

	# and copy the smoothing to the other models
	lam = gam50.lam

	# fit a few more models
	gam95 = ExpectileGAM(expectile=0.95, lam=lam).fit(X, y)
	gam75 = ExpectileGAM(expectile=0.75, lam=lam).fit(X, y)
	gam25 = ExpectileGAM(expectile=0.25, lam=lam).fit(X, y)
	gam05 = ExpectileGAM(expectile=0.05, lam=lam).fit(X, y)
	
	XX = gam50.generate_X_grid(term=0, n=100)
	XX95 = list(gam95.predict(XX).flatten())
	XX75 = list(gam75.predict(XX).flatten())
	XX50 = list(gam50.predict(XX).flatten())
	XX25 = list(gam25.predict(XX).flatten())
	XX05 = list(gam05.predict(XX).flatten())
	XX = list(XX.flatten())

	fig = px.scatter(dfRef, x='Age', y=selVar, opacity=0.2)
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX95, marker=dict(color='MediumPurple', size=10), name='perc95'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX75, marker=dict(color='Orchid', size=10), name='perc75'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX50, marker=dict(color='MediumVioletRed', size=10), name='perc50'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX25, marker=dict(color='Orchid', size=10), name='perc25'))
	fig.add_trace( go.Scatter(mode='lines', x=XX, y=XX05, marker=dict(color='MediumPurple', size=10), name='perc05'))
	fig.add_trace( go.Scatter(
		mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(),marker=dict(color='Red', size=16,line=dict( color='MediumPurple', width=2)), name='Sub'))
	
	fig.write_html(fname, include_plotlyjs = 'cdn')
	
def plotToHtml(dfRef, dfSub, selVar, fname):

	x = dfRef.Age.values.tolist()
	y = dfRef[selVar].values.tolist()
	
	fig = px.scatter(dfRef, x='Age', y=selVar)
	fig.add_trace( go.Scatter(
		mode='markers', x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(),marker=dict(color='Red', size=16,line=dict( color='MediumPurple', width=2))))
	#fig = px.scatter(x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), color=['Red'])
	
	#px.scatter(x=dfSub.Age.tolist(), y=dfSub[selVar].tolist(), color="species")
	
	fig.write_html(fname, include_plotlyjs = 'cdn')

	#plt.plot(dfSub.Age, dfSub[selVar], 'ro', markersize=10)
	#plt.xlabel('Age')
	#plt.ylabel(selVar)
	

def writeHtml(htmlName1, table1, outName):

	html_string = '''
	<html>
	<head>
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
		<style>body{ margin:0 100; background:whitesmoke; }</style>
	</head>
	<body>
		<h1>MRI Report of the Subject</h1>

		<!-- *** Section 1 *** --->
		<h2>Section 1: Age plot</h2>
		<iframe width="1000" height="550" frameborder="0" seamless="seamless" scrolling="no" \
	src="''' + htmlName1 + '''"></iframe>
		<p>GM Age trends.</p>
		
		<h3>Reference table: </h3>
		''' + table1 + '''
	</body>
	</html>'''

	f = open(outName,'w')
	f.write(html_string)
	f.close()

################################################ END OF FUNCTIONS ################################################
	
############## MAIN ##############
#DEF
def _main():

	##########################################################################
	##### Read reference ROI values
	dfRef = pd.read_csv(MUSE_Ref_Values).dropna()

	##########################################################################
	##### Read subject data (demog and MRI)
	## Args
	FLAGS,parser = read_flags()

	## Read info
	subjAge = None
	subjSex = None
	if FLAGS.info is not None:
		d={}
		with open(FLAGS.info) as f:
			reader = _csv.reader( f )
			subDict = {rows[0]:rows[1] for rows in reader}

	## Read bmask
	bmaskVol = None
	if FLAGS.bmask is not None:
		bmaskVol = calcMaskVolume(FLAGS.bmask)
		print('bmask : ' + str(bmaskVol))

	## Read icv
	icvVol = None
	if FLAGS.icv is not None:
		icvVol = calcMaskVolume(FLAGS.icv)
		print('icv : ' + str(icvVol))

	## Read roi
	roiVols = None
	if FLAGS.roi is not None:
		roiVols = calcRoiVolumes(FLAGS.roi, MUSE_ROI_Mapping)

	## Create subject dataframe with all input values
	dfSub = pd.DataFrame(columns=dfRef.columns)

	dfSub.loc[0,'MRID'] = subDict['MRID']
	dfSub.loc[0,'Age'] = float(subDict['Age'])
	dfSub.loc[0,'Sex'] = subDict['Sex']

	dfSub.DLICV = icvVol
	
	dfSub.MUSE_Volume_702 = bmaskVol
	
	for tmpRoi in SEL_ROI:
		if tmpRoi.replace('MUSE_Volume_','') in roiVols:
			dfSub.loc[0, tmpRoi] = roiVols[tmpRoi.replace('MUSE_Volume_','')]
		
	##########################################################################
	##### Rename ROIs
	dictMUSE = dict(zip(SEL_ROI,SEL_ROI_Rename))
	dfRef = dfRef.rename(columns=dictMUSE)
	dfSub = dfSub.rename(columns=dictMUSE)

	##########################################################################
	##### ICV correct MUSE values
	
	## Correct ref values
	dfRefTmp = dfRef[dfRef.columns[dfRef.columns.str.contains('MUSE_')]]
	dfRefTmp = dfRefTmp.div(dfRef.DLICV, axis=0)*dfRef.DLICV.mean()
	dfRefTmp = dfRefTmp.add_suffix('_ICVCorr')
	dfRef = pd.concat([dfRef, dfRefTmp], axis=1)
	print(dfRef.loc[0])

	## Correct sub values
	dfSubTmp = dfSub[dfSub.columns[dfSub.columns.str.contains('MUSE_')]]
	dfSubTmp = dfSubTmp.div(dfSub.DLICV, axis=0)*dfRef.DLICV.mean()
	dfSubTmp = dfSubTmp.add_suffix('_ICVCorr')
	dfSub = pd.concat([dfSub, dfSubTmp], axis=1)
	print(dfSub.loc[0])


	##########################################################################
	##### Plot values

	### Plot icv
	#selVar = 'DLICV'
	#dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plotWithRef(dfSel, dfSub, selVar)

	#### Plot bmask
	#selVar = 'MUSE_TotalBrain'
	#dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plotWithRef(dfSel, dfSub, selVar)

	#### Plot gm
	#selVar = 'MUSE_GM'
	#dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plotWithRef(dfSel, dfSub, selVar)

	#### Plot bmask ICVCorr
	#selVar = 'MUSE_TotalBrain_ICVCorr'
	#dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plotWithRef(dfSel, dfSub, selVar)

	#### Plot gm ICVCorr
	#selVar = 'MUSE_GM_ICVCorr'
	#dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plotWithRef(dfSel, dfSub, selVar)
	
	#dfTmp = dfSel[['Age',selVar]].head(100)
	#dfTmp.to_csv('tmpCsv.csv', index=False)
	
	### Plot gm ICVCorr
	htmlName1 = './tmp1.html'
	selVar = 'MUSE_GM_ICVCorr'
	dfSel = dfRef[dfRef.Sex == subDict['Sex']]
	#plot1 = plotToHtml(dfSel, dfSub, selVar, htmlName1)
	plot1 = plotWithRef(dfSel, dfSub, selVar, htmlName1)

	table1 = dfSub.describe()
	table1 = table1.to_html().replace('<table border="1" class="dataframe">','<table class="table table-striped">') # use bootstrap styling
	HTML(table1)

	writeHtml(htmlName1, table1, './report1.html')

#IF
if __name__ == '__main__':
	_main()
#ENDIF
