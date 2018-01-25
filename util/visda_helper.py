## VisDa Labels

from collections import namedtuple
import numpy as np

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

Label = namedtuple( 'Label' , [

	'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
					# We use them to uniquely name a class

	'id'          , # An integer ID that is associated with this label.
					# The IDs are used to represent the label in ground truth images
					# An ID of -1 means that this label does not have an ID and thus
					# is ignored when creating ground truth images (e.g. license plate).
					# Do not modify these IDs, since exactly these IDs are expected by the
					# evaluation server.

	'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
					# ground truth images with train IDs, using the tools provided in the
					# 'preparation' folder. However, make sure to validate or submit results
					# to our evaluation server using the regular IDs above!
					# For trainIds, multiple labels might have the same ID. Then, these labels
					# are mapped to the same class in the ground truth images. For the inverse
					# mapping, we use the label that is defined first in the list below.
					# For example, mapping all void-type classes to the same ID in training,
					# might make sense for some approaches.
					# Max value is 255!

	'category'    , # The name of the category that this label belongs to

	'categoryId'  , # The ID of this category. Used to create ground truth images
					# on category level.

	'color'       , # The color of this label
] )

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

num_classes = 20
ignore_labels = [0]

shape = (1052, 1914)

img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
img_stdev = np.array([58.96662254, 59.37453629, 60.42706421])

class_weights = [
		0.11015086754409116, 0.3215191428487277, 0.16931354687873518,
		0.13652687513867484, 0.08374657046653439, 0.0751681507220292,
		0.025465338440762618, 0.020420402719133743, 0.01830169146084464,
		0.012004842302352221, 0.010523853930535492, 0.006177791836249105,
		0.0036909182443827914, 0.00360455521521489, 0.0013540632950341603,
		0.0008573018347492742, 0.000549278449228592, 0.00029019466378333284,
		0.0002823316110694899, 0.00005228239786719245
]

#--------------------------------------------------------------------------------
# List of labels
#--------------------------------------------------------------------------------

labels = [
	#		name				id		trainID	category		catID	color
	Label(	"unlabeled",		0,		0,		'void',			0,		(0,0,0)			),
	Label(	"road",				1,		1,		'flat',			1,		(128,64,128)	),
	Label(	"building",			2,		2,		'construction',	2,		(70,70,70)		),
	Label(	"sky",				3,		3,		'sky',			5,		(180,130,70)	),
	Label(	"sidewalk",			4,		4,		'flat',			1,		(232,35,244)	),
	Label(	"vegetation",		5,		5,		'nature',		4,		(35,142,107)	),
	Label(	"car",				6,		6,		'vehicle',		7,		(142,0,0)		),
	Label(	"terrain",			7,		7,		'nature',		4,		(152,251,152)	),
	Label(	"wall",				8,		8,		'construction',	2,		(156,102,102)	),
	Label(	"truck",			9,		9,		'vehicle',		7,		(70,0,0)		),
	Label(	"pole",				10,		10,		'object',		3,		(153,153,153)	),
	Label(	"fence",			11,		11,		'construction',	2,		(153,153,190)	),
	Label(	"person",			12,		12,		'human',		6,		(60,20,220)		),
	Label(	"bus",				13,		13,		'vehicle',		7,		(100,60,0)		),
	Label(	"traffic_light",	14,		14,		'object',		3,		(30,170,250)	),
	Label(	"traffic_sign",		15,		15,		'object',		3,		(0,220,220)		),
	Label(	"train",			16,		16,		'vehicle',		7,		(100,80,0)		),
	Label(	"rider",			17,		17,		'human',		6,		(0,0,255)		),
	Label(	"motorcycle",		18,		18,		'vehicle',		7,		(230,0,0)		),
	Label(	"bicycle",			19,		19,		'vehicle',		7,		(32,11,119)		),
]


#--------------------------------------------------------------------------------
# Dictionaries for lookup
#--------------------------------------------------------------------------------

# name to label object
name2label		= { label.name    : label for label in labels           }

# id to label object
id2label		= { label.id     : label for label in labels           }

# trainId to label object
trainId2label	= { label.trainId : label for label in reversed(labels) }

# category to list of label objects
category2labels = {}
for label in labels:
	category = label.category
	if category in category2labels:
		category2labels[category].append(label)
	else:
		category2labels[category] = [label]

