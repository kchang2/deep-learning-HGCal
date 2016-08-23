import argparse
import ROOT as rt
import numpy as np
import os, sys, platform
from copy import deepcopy
import glob

def get_all_indices(list, item):
     return [index for index in xrange(len(list)) if list[index] == item]

def checkEqual(lst):
	return lst[1:] == lst[:-1]

def getThicknesses(outdir):
	''' Returns an organized list with wafers grouped by thickness.
		Parameters
		----------
		outdir :	path to list of .npy files

		Returns
		-------
		list of thicknesses in [[100um], [200um], [300um]] format

	'''
	# change directory
	os.chdir(outdir)
	files = [f for f in os.listdir('.') if f.endswith('.npy')]
	# files = [f for f in files if '12' not in f]

	thickness_list = [[] for i in range(3)] 	# 100 um, 200 um, 300 um (just for layer 10)

	for fi in files:
		arr = np.load(f)

		for event in arr:
			for hit in event[0]:
				if int(hit['layer']) == 10 and not any(hit['wafer'] in sublist for sublist in thickness_list):  	# only record in layer 10
						thickness_list[int(hit['thickness'])/100 - 1].append(int(hit['wafer']))
				if int(hit['layer']) == 10  and hit['wafer'] == 0:
					print 'wtf, 0 wafer', hit

	for thick in thickness_list:
		print sorted(thick)

	flattened_thickness_list = sorted([y for x in thickness_list for y in x])
	print 'flattened = ', flattened_thickness_list
	print 'length = ', len(flattened_thickness_list)
	ls = range(max(flattened_thickness_list))

	print ls == flattened_thickness_list
	print 'difference = ', sorted(list(set(ls).difference(flattened_thickness_list)))

	return thickness_list



def genThickness():
	''' Give thickness based on precalculated results.
		CMSSW 8.1, all layers contain same positional wafers,
		so took all the wafers from all layers (essentially largest layer).
	'''
	t100um = [116, 117, 118, 136, 137, 138, 139, 140, 141, 157, 158, 159, 160, 161, 162, 163, 164, 165, \
			180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 203, 204, 205, 206, 207, 208, 209, 210, 224, \
			225, 226, 227, 228, 229, 243, 244, 245, 246, 247, 248, 262, 263, 264, 265, 266, 267, 281, 282, \
			283, 284, 285, 286, 287, 288, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 326, 327, 328, \
			329, 330, 331, 332, 333, 334, 350, 351, 352, 353, 354, 355, 373, 374, 375]

	t200um = [39, 40, 41, 42, 43, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 75, 76, 77, 78, 79, \
			80, 81, 82, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 110, 111, 112, 113, 114, \
			115, 119, 120, 121, 122, 123, 124, 131, 132, 133, 134, 135, 142, 143, 144, 145, 146, 153, 154, \
			155, 156, 166, 167, 168, 169, 176, 177, 178, 179, 190, 191, 192, 193, 200, 201, 202, 211, 212, \
			213, 220, 221, 222, 223, 230, 231, 232, 233, 239, 240, 241, 242, 249, 250, 251, 252, 258, 259, \
			260, 261, 268, 269, 270, 271, 278, 279, 280, 289, 290, 291, 298, 299, 300, 301, 312, 313, 314, \
			315, 322, 323, 324, 325, 335, 336, 337, 338, 345, 346, 347, 348, 349, 356, 357, 358, 359, 360, \
			367, 368, 369, 370, 371, 372, 376, 377, 378, 379, 380, 381, 388, 389, 390, 391, 392, 393, 394, \
			395, 396, 397, 398, 399, 400, 401, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 428, \
			429, 430, 431, 432, 433, 434, 435, 436, 437, 448, 449, 450, 451, 452]

	t300um = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
			26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 64, \
			65, 66, 67, 68, 69, 70, 71, 83, 84, 85, 86, 87, 88, 89, 104, 105, 106, 107, 108, 109, 125, 126, \
			127, 128, 129, 130, 147, 148, 149, 150, 151, 152, 170, 171, 172, 173, 174, 175, 194, 195, 196, \
			197, 198, 199, 214, 215, 216, 217, 218, 219, 234, 235, 236, 237, 238, 253, 254, 255, 256, 257, \
			272, 273, 274, 275, 276, 277, 292, 293, 294, 295, 296, 297, 316, 317, 318, 319, 320, 321, 339, \
			340, 341, 342, 343, 344, 361, 362, 363, 364, 365, 366, 382, 383, 384, 385, 386, 387, 402, 403, \
			404, 405, 406, 407, 408, 420, 421, 422, 423, 424, 425, 426, 427, 438, 439, 440, 441, 442, 443, \
			444, 445, 446, 447, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, \
			468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, \
			487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, \
			506, 507, 508, 509, 510, 511, 512, 513, 514, 515]

	return [t100um, t200um, t300um]




