# Cityscapes labels
#

from collections import namedtuple
import numpy as np


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
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

# shape = (1052, 1914)
shape = (1024, 2048)

img_mean = np.array([122.67892 , 116.66877, 104.00699])
img_stdev = 60


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
	#       name                     id    trainId   category            catId    color
	Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0      , (  0,  0,  0) ),
	Label(  'road'                 ,  7 ,        1 , 'flat'            , 1      , (128, 64,128) ),
	Label(  'sidewalk'             ,  8 ,        4 , 'flat'            , 1      , (244, 35,232) ),
	Label(  'building'             , 11 ,        2 , 'construction'    , 2      , ( 70, 70, 70) ),
	Label(  'wall'                 , 12 ,        8 , 'construction'    , 2      , (102,102,156) ),
	Label(  'fence'                , 13 ,       11 , 'construction'    , 2      , (190,153,153) ),
	Label(  'pole'                 , 17 ,       10 , 'object'          , 3      , (153,153,153) ),
	Label(  'traffic light'        , 19 ,       14 , 'object'          , 3      , (250,170, 30) ),
	Label(  'traffic sign'         , 20 ,       15 , 'object'          , 3      , (220,220,  0) ),
	Label(  'vegetation'           , 21 ,        5 , 'nature'          , 4      , (107,142, 35) ),
	Label(  'terrain'              , 22 ,        7 , 'nature'          , 4      , (152,251,152) ),
	Label(  'sky'                  , 23 ,        3 , 'sky'             , 5      , ( 70,130,180) ),
	Label(  'person'               , 24 ,       12 , 'human'           , 6      , (220, 20, 60) ),
	Label(  'rider'                , 25 ,       17 , 'human'           , 6      , (255,  0,  0) ),
	Label(  'car'                  , 26 ,        6 , 'vehicle'         , 7      , (  0,  0,142) ),
	Label(  'truck'                , 27 ,        9 , 'vehicle'         , 7      , (  0,  0, 70) ),
	Label(  'bus'                  , 28 ,       13 , 'vehicle'         , 7      , (  0, 60,100) ),
	Label(  'train'                , 31 ,       16 , 'vehicle'         , 7      , (  0, 80,100) ),
	Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7      , (  0,  0,230) ),
	Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7      , (119, 11, 32) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }

# category to list of label objects
category2labels = {}
for label in labels:
	category = label.category
	if category in category2labels:
		category2labels[category].append(label)
	else:
		category2labels[category] = [label]
