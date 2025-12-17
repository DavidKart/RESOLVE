import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator
import datetime
import os	
import json
from scipy.signal.windows import gaussian
import math
from scipy import ndimage
import ctypes


def iterateBoxesWindows(collapseWindow_i, localResMap_out, boxes_iterate, dimension, windows, window_size_i, blueprint_box, sizeMap_padded, boxSize, resolutions, filterChoice, apix, slicesPadding, padded_inputMap_1, padded_inputMap_2, res_obj, res_obj_inv, freqMap, shells, falloff, gpu_ids, runOnGPU, it_randomMaps, partial_locaRes, printDebugging, corrected_box_size, maxWindow_half, stepSize, referenceDistSize, numCores, localResMap_size, p_cutoff, test2, lowRes, partial_fillMap, signalMask_padded, mask_measure, config, outputDir, runOnAveragedMap, preName = ""):
	"""
	This is the primary function called for local resolution measurements.
 	Including bandpass filtering, local correlation calculations, q-value calculatoins, 
  	and median (global) resolution calculations.
	"""	

	# Some initializing
	import multiprocessing as mp
	queue = mp.Queue()
	box_count = 0
	dictsTiltSeries = []
	merged_dictsTiltSeries = {}
	overallValues = []
	actualVAllues = []
  
	# This loop is outdated, only kept for potential later usage. There is only one box.
	for box_it in boxes_iterate: 
		dictTiltSeries = {}
		j,k,l = box_it[0], box_it[1], box_it[2]
		start_box = datetime.datetime.now()
		box_count += 1
		print("Running on " + str(numCores) + " core(s)")

		# Calculate borders for input map
		borderInput_0, borderInput_1, borderInput_2 = [], [], []
		borderInput_0 = [j*corrected_box_size[0], np.min([sizeMap_padded[0], (2*maxWindow_half[0])+((j+1)*corrected_box_size[0])])]
		borderInput_1 = [k*corrected_box_size[1], np.min([sizeMap_padded[1], (2*maxWindow_half[1])+((k+1)*corrected_box_size[1])])]
		if dimension == 3: borderInput_2 = [l*corrected_box_size[2], np.min([sizeMap_padded[2], (2*maxWindow_half[2])+((l+1)*corrected_box_size[2])])]
		bordersInput = [borderInput_0, borderInput_1, borderInput_2][:dimension]
										
		# Calculate borders for current map    
		borderCurr_0, borderCurr_1, borderCurr_2 = [], [], []       
		borderCurr_0 = [0, np.min([boxSize[0], sizeMap_padded[0]-j*corrected_box_size[0]])]
		borderCurr_1 = [0, np.min([boxSize[1], sizeMap_padded[1]-k*corrected_box_size[1]])]
		if dimension == 3: borderCurr_2 = [0, np.min([boxSize[2], sizeMap_padded[2]-l*corrected_box_size[2]])]                
		bordersCurr = [borderCurr_0, borderCurr_1, borderCurr_2][:dimension]

		# Create slicer and crop
		slicesInput = [slice(i[0], i[1]) for i in bordersInput]
		slicerCurr = [slice(i[0], i[1]) for i in bordersCurr]
		currentMap1 = np.copy(blueprint_box)
		currentMap2 = np.copy(blueprint_box)
		currentMap1[tuple(slicerCurr)] = padded_inputMap_1[tuple(slicesInput)]
		currentMap2[tuple(slicerCurr)] = padded_inputMap_2[tuple(slicesInput)]

		if signalMask_padded is not None:
			signal_maskBox = np.zeros(blueprint_box.shape, dtype=bool)
			signal_maskBox[tuple(slicerCurr)] = signalMask_padded[tuple(slicesInput)]

		if collapseWindow_i: # For tilt-series
			fft_map1_ini = []
			fft_map2_ini = []
			for currPlane in range(blueprint_box.shape[0]):
				fft_map1_ini.append(np.copy(res_obj(currentMap1[currPlane])))
				fft_map2_ini.append(np.copy(res_obj(currentMap2[currPlane])))
			fft_map1_ini = np.array(fft_map1_ini)
			fft_map2_ini = np.array(fft_map2_ini)
		else:
			fft_map1_ini = np.copy(res_obj(currentMap1))
			fft_map2_ini = np.copy(res_obj(currentMap2))

		currentMaps4_fft = []
		currentMap4 = []
		for _ in range(it_randomMaps):
			# For anything but refined maps, go for phase permutation as reference
			if not runOnAveragedMap:
				angles = np.angle(fft_map2_ini)
				angles_flat = angles.ravel()
				np.random.shuffle(angles_flat)
				currentMaps4_fft.append(np.abs(fft_map2_ini) * np.exp(1j * angles))    
	
			else:
				currentMap4 = np.random.choice(padded_inputMap_2[tuple(slicesPadding)].flatten(), size=np.prod(blueprint_box.shape)).reshape(blueprint_box.shape)
				# For tilts-series, treat tilts as 2D images
				if collapseWindow_i:
					currentMaps4_fft_curr = []
					for currPlane in range(blueprint_box.shape[0]):
						currentMaps4_fft_curr.append(np.copy(res_obj(currentMap4[currPlane])))
					currentMaps4_fft.append(np.copy(currentMaps4_fft_curr))

				else:
					currentMaps4_fft.append(np.copy(res_obj(currentMap4)))			

		# Clean
		del currentMap1
		del currentMap2

  
		locResMap = [[] for _ in range(len(windows))]
		WinCountNumCore = 0
		res = []
 
		# Iterate over all windows
		for index_i, i in enumerate(windows):
			if runOnGPU >= 1:
				from numba import cuda
				with cuda.gpus[gpu_ids[0]]:
					freqMapCuda = cuda.to_device(freqMap)
			else:
				freqMapCuda = freqMap
 
			WinCountNumCore += 1
			curr_res = resolutions[index_i]
			windowSize = i
			print("Calculations for resolution " + str(1/curr_res))
   
   
			# Bandpass filtering
			permutated_sample1_filtered = []
			permutated_sample2_filtered = []
			startCreation = datetime.datetime.now()
			if collapseWindow_i: # Tilt-series
				bandpassFilter = filterChoice(apix, fft_map1_ini[0], res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], True)
				sample1_filtered = filterChoice(apix, fft_map1_ini, res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter, True)
				sample2_filtered = filterChoice(apix, fft_map2_ini, res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter, True)
				for ranInd in range(it_randomMaps): # If multiple randomly permutated maps are needed to create a large enough reference distribution
					permutated_sample2_filtered.append(filterChoice(apix, currentMaps4_fft[ranInd], res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter, True))    
				sample1_filtered = np.array(sample1_filtered)
				sample2_filtered = np.array(sample2_filtered)
				permutated_sample2_filtered = np.array(permutated_sample2_filtered)
			else:
				bandpassFilter = filterChoice(apix, fft_map1_ini, res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], True)
				sample1_filtered = filterChoice(apix, fft_map1_ini, res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter)
				sample2_filtered = filterChoice(apix, fft_map2_ini, res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter) 
				for ind_rand in range(it_randomMaps): # If multiple randomly permutated maps are needed to create a large enough reference distribution
					permutated_sample2_filtered.append(filterChoice(apix, currentMaps4_fft[ind_rand], res_obj_inv, freqMapCuda, shells[index_i][0], shells[index_i][1], falloff, gpu_ids[0], False, bandpassFilter))
			# print("full creation taking " + str(datetime.datetime.now()-startCreation) + "\n")
			permutated_sample1_filtered = [sample1_filtered]

			# GPU correlation measurements
			if runOnGPU >= 1:
				from numba import cuda
				for device_id in (gpu_ids):
					with cuda.gpus[device_id]:
						cuda.current_context().reset()
				start_GPU = datetime.datetime.now()
				locResMap[index_i] = localResolutionsGPU(collapseWindow_i, sample1_filtered, sample2_filtered, permutated_sample2_filtered, permutated_sample1_filtered, windowSize, window_size_i, index_i, corrected_box_size, maxWindow_half, stepSize, gpu_ids, bordersCurr, it_randomMaps, referenceDistSize)
				for device_id in (gpu_ids):
					with cuda.gpus[device_id]:
						cuda.current_context().reset()
				# print("full localResolutionsGPU taking " + str(datetime.datetime.now()-start_GPU) + "\n")

				if printDebugging: # Debugging options
					import mrcfile
					compareMap = np.zeros(locResMap[index_i].shape)
					compareMap[locResMap[index_i]<0.01] = 1
					binaryResMap = mrcfile.new(os.path.join("debugging", "checkTresholdedMap", preName + "_res" + str(curr_res) + "_" "win" + str((windowSize)*apix) + ".mrc"), overwrite=True)
					binaryResMap.set_data(np.float32(compareMap))
					binaryResMap.voxel_size = apix
					binaryResMap.close() 
					del	compareMap
 
			# CPU correlation measurements
			else:
				proc = mp.Process(target=partial_locaRes, args=(queue, collapseWindow_i, sample1_filtered, sample2_filtered, permutated_sample2_filtered, permutated_sample1_filtered, windowSize, window_size_i, index_i, bordersCurr, it_randomMaps, referenceDistSize))
				proc.start()
				if (WinCountNumCore >= numCores) or ((index_i) >= (len(windows)-1)):
					for _ in range(WinCountNumCore):
						output_queue = queue.get()
						locResMap[output_queue[0]] = output_queue[1]
						# pValsDistRes[output_queue[0]] = output_queue[2][0]
					queue = mp.Queue()
					WinCountNumCore = 0
					# print("finished calculating for " + str(index_i+1) + " of " + str(len(windows)) + " windows")
			res.append(curr_res)


		end_box = datetime.datetime.now()
		# print("finished box taking " + str(end_box-start_box) + "\n")

		# Map filling (From per-shell maps with p-values to one q-thresholded resolution map)
		numPval = len(resolutions)
		locResMap = np.array(locResMap)
		filled = False
		if (runOnGPU >= 1):
			try:
				print("Try map filling on GPU")
				localResMap_out = fillMap_cuda(j, k, l, localResMap_size, localResMap_out, locResMap, p_cutoff, test2, res, lowRes, numPval, gpu_ids[0])
				filled = True
			except:
				filled = False
		if not filled:
			print("Map filling on CPU")
			locResMap = np.array(locResMap, dtype=np.float32)
			queue = mp.Queue()
			itRange = int(np.ceil(localResMap_size[0]/numCores))
			processes = []
   
			#prepare arrays to make accessible for multithreading
			shapeLocResOut = localResMap_out.shape
			shared_base_locResOut = mp.Array(ctypes.c_float, localResMap_out.size)
			shared_np_locResOut = np.frombuffer(shared_base_locResOut.get_obj(), dtype=localResMap_out.dtype).reshape(shapeLocResOut)
			np.copyto(shared_np_locResOut, localResMap_out)
   
			shapeLocRes = locResMap.shape
			shared_base_locResMap = mp.Array(ctypes.c_float, locResMap.size)
			shared_np_locResMap = np.frombuffer(shared_base_locResMap.get_obj(), dtype=locResMap.dtype).reshape(shapeLocRes)
			np.copyto(shared_np_locResMap, locResMap)
   
			for i_fill in range(numCores):
				shape0Range = [i_fill*itRange, (i_fill+1)*itRange]
				# proc = mp.Process(target=partial_fillMap, args=(queue, j, k, l, shape0Range, localResMap_size, shared_base_locResOut, shared_base_locResMap, shapeLocResOut, shapeLocRes, res))
				proc = mp.Process(target=partial_fillMap, args=(queue, j, k, l, shape0Range, localResMap_size, shared_np_locResOut, shared_np_locResMap, res))
				proc.start()
				processes.append(proc)
	   
			for proc in processes:
				proc.join()
	
			localResMap_out = np.array(shared_np_locResOut)
   
		# print("finished filling " + str(datetime.datetime.now()-start_box) + "\n")

		# Towards median (global) resolution estimation (does not apply for refined maps)
		if not runOnAveragedMap:
			if signalMask_padded is not None:
				signal_maskBox = np.array(signal_maskBox)
				signal_maskBox = signal_maskBox[maxWindow_half[0]:-maxWindow_half[0], maxWindow_half[1]:-maxWindow_half[1]]
		
				if dimension == 2:
					signal_maskBox = signal_maskBox[::stepSize[0], ::stepSize[1]]
				if dimension == 3:
					signal_maskBox = signal_maskBox[::stepSize[0], ::stepSize[1], ::stepSize[2]]      

			lowResRounded = int(lowRes*100)/100
			resList_glob = []

			slicesLocResMapOut = [slice(box_it[i_box]*localResMap_size[i_box], (1+box_it[i_box])*localResMap_size[i_box]) for i_box in range(dimension)]
			localResMap_outTemp = localResMap_out[tuple(slicesLocResMapOut)]
			
			if dimension == 2:
				if signalMask_padded is not None:
					signal_maskBox = signal_maskBox[:localResMap_outTemp.shape[0], :localResMap_outTemp.shape[1]]
				locResMap = locResMap[:, :localResMap_outTemp.shape[0], :localResMap_outTemp.shape[1]]
			if dimension == 3:
				locResMap = locResMap[:, :localResMap_outTemp.shape[0], :localResMap_outTemp.shape[1], :localResMap_outTemp.shape[2]]
				if signalMask_padded is not None:
					signal_maskBox = signal_maskBox[:localResMap_outTemp.shape[0], :localResMap_outTemp.shape[1], :localResMap_outTemp.shape[2]]
			overallValues.append(np.prod(locResMap.shape))


			for index_i in range(len(locResMap)):
				res_rounded = int((1/res[index_i])*100)/100
				if res_rounded not in dictTiltSeries:
					resList_glob.append(res_rounded)
					dictTiltSeries[res_rounded] = []
				if collapseWindow_i or (dimension == 3):
					# print("MEASURE GLOBAL RES PER SLICE")
					for z_slice in range(len(locResMap[index_i])):
						pValues_zSlice = locResMap[index_i][z_slice] 
						if signalMask_padded is not None:
							# if collapseWindow_i: # TODO
							# 	pValues_zSlice = pValues_zSlice.flatten()
							# else:
							pValues_zSlice = pValues_zSlice[signal_maskBox[z_slice] == 1]
						else:
							pValues_zSlice = pValues_zSlice[localResMap_outTemp[z_slice] < lowResRounded]
						actualVAllues.append(np.prod(pValues_zSlice.shape))
						if mask_measure == "median":
							median_pValue_zSlice = np.median(pValues_zSlice.flatten())
						if mask_measure == "average":
							median_pValue_zSlice = np.average(pValues_zSlice.flatten())
						dictTiltSeries[res_rounded].append(float(median_pValue_zSlice))
				else:
					pValues = locResMap[index_i]
					if signalMask_padded is not None:
						pValues = pValues[signal_maskBox == 1]
					else:
						pValues = pValues[localResMap_outTemp < lowResRounded]
					# pValues = pValues[(localResMap_outTemp < lowResRounded)]
					if mask_measure == "median":
						dictTiltSeries[res_rounded].append(float(np.median(pValues.flatten())))
					if mask_measure == "average":
						dictTiltSeries[res_rounded].append(float(np.average(pValues.flatten())))
					actualVAllues.append(np.prod(pValues.shape))

			dictsTiltSeries.append(dictTiltSeries)

	# Finalising median (global) resolution estimates (does not apply for refined maps)
	if not runOnAveragedMap:
		pValListGlobal = []
		# in case measurement for multiple boxes, we need concat, else, we do not.
		merged_dictsTiltSeries = combine_concat(dictsTiltSeries)

		# finally, iterate through resolutions to collapse all median planes resolutions into one value
		for i in merged_dictsTiltSeries:
			# median_p_per_res = np.median(merged_dictsTiltSeries[i])
			# MEAN OR MEDIAN???
			median_p_per_res = np.mean(merged_dictsTiltSeries[i])
			pValListGlobal.append(median_p_per_res)

  
		ratioSignal = np.round(np.sum(actualVAllues)/np.sum(overallValues), 2)
		pVals_qual, actualRes_global_new = getFittedResolution(p_cutoff, resList_glob, pValListGlobal, lowResRounded)
		if collapseWindow_i or (dimension == 3): 
			plot_heatmap_pvalue(merged_dictsTiltSeries, os.path.join(outputDir, preName + "_pValuePlot"), 0, 0.05, "Slices", "Resolution", "p-Value", 7, 4, "svg", actualRes_global_new, ratioSignal)

		plot_heatmap_qvalue(resList_glob, pValListGlobal, os.path.join(outputDir, preName + "_qValuePlot"), 0, 0.5, "1/Resolution", "q-value", 8, 5, "svg", actualRes_global_new, ratioSignal)
		print(str(mask_measure) + " resolution calculated in signal regions: " + str(actualRes_global_new))
		print("ratio of considered signal regions: " + str(ratioSignal))  
		with open(os.path.join(outputDir, preName + "_qualityPlot.json"), "w") as json_file:
			json.dump(merged_dictsTiltSeries, json_file, indent=4)
	else:
		actualRes_global_new, ratioSignal = 0,0

	return localResMap_out, actualRes_global_new, ratioSignal


def combine_concat(dict_list):
	"""
	Towards median resolution estimation. This is in order to combine values over multiple boxes (in case map is too large and need to be split to be processed.)
	This is an irrelevant extra functionality not used right now we we have only one box, kept only for potential later use.
	"""
	result = {}
	for d in dict_list:
		for key, values in d.items():
			if key not in result:
				result[key] = []
			result[key].extend(values)
	return result


def getFittedResolution(p_cutoff, x_list, y_list, lowResRounded, num_samples=100):
	"""   
    Interpolates the Fourier Shell Correlation (FSC) curve and applies 
    Benjamini-Yekutieli FDR correction to determine the resolution cutoff 
    at p=0.05 significance level.
    """

	from scipy.interpolate import interp1d

	# Ensure x and y are numpy arrays
	x = np.array(x_list)
	y = np.array(y_list)

	# Create interpolation function
	interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')

	# Generate regular x-values for interpolation (num_samples)
	x_min = np.min(x)
	x_max = np.max(x)
	sampled_x = np.linspace(x_min, x_max, num_samples)

	# Interpolate y-values at regular x intervals
	interpolated_y = interp_func(sampled_x)
	qVals_FDR = pAdjust_BY(np.array(interpolated_y[::-1]))

	res_index = calcResIndex(qVals_FDR, 0.05, False) #p 0.05 for median resolution
	# print(res_index)
	if res_index < 0:
		actualRes_global = lowResRounded
	else:
		actualRes_global = int((sampled_x[::-1][res_index]) * 100) / 100  # Round to 2 decimal places
	return 1-np.mean(qVals_FDR), actualRes_global


def pAdjust_BY(pValues):
	"""   
	CPU function for Benjamini-Yekutieli FDR correction to determine the resolution cutoff.
    
    """

	numPVal = len(pValues)

	pSortInd = np.argsort(pValues)
	pSort = pValues[pSortInd]

	pAdjust = np.zeros(numPVal)
	prevPVal = 1.0

	#use expansion for harmonic series
	Hn = math.log(numPVal) + 0.5772 + 0.5/numPVal - 1.0/(12*numPVal**2) + 1.0/(120*numPVal**4)

	for i in range(numPVal-1, -1, -1):
		pAdjust[i] =  min(prevPVal, pSort[i]*(numPVal/(i+1.0))*Hn)
		prevPVal = pAdjust[i]

	pSortIndOrig = np.argsort(pSortInd)
	return pAdjust[pSortIndOrig]


def writeMrcFile(data, name, apix):
	"""
	Writing mrc file.
	"""
	import mrcfile
	mrcMap = mrcfile.new(name, overwrite=True)
	data = np.float32(data)
	mrcMap.set_data(data)
	mrcMap.voxel_size = apix
	mrcMap.close()

def plot_heatmap_qvalue(x_values, y_values, output_path, minV, maxV, xAxisLabel, yAxisLabel, figSizeX=10, figSizeY=4, format="png", actualResGlobal = 0, ratioSignal=0):
	"""
	Create what is refered to as a q-value plot in the paper from which the median resolution is derived 
 	for tomograms, tilt-series and micrographs.
	"""

	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	x_values = np.array(x_values, dtype=np.float32) # Resolutions
	x_values = 1/x_values
	y_values = np.array(y_values, dtype=np.float32)  # p-values

	# Create the plot
	plt.figure(figsize=(figSizeX, figSizeY))
	plt.rcParams['font.size'] = 16

	plt.plot(x_values, y_values, linestyle='-', marker='o', color='b', markersize=3)
	plt.xlabel(xAxisLabel)
	plt.ylabel(yAxisLabel)
	plt.ylim(minV, maxV)  
	plt.grid(False)
	plt.title("median resolution of " + str(actualResGlobal) +  " within signal (signal ratio: " + str(ratioSignal) + ")")
	plt.tight_layout()

	# Save the plot in the specified format
	if format == "svg":
		plt.rcParams["svg.fonttype"] = "none"
		plt.savefig(output_path + ".svg", format="svg")
	elif format == "pdf":
		plt.rcParams["svg.fonttype"] = "none"
		plt.savefig(output_path + ".pdf", format="pdf")
	else:
		plt.savefig(output_path + ".png")
	plt.close()


def plot_heatmap_pvalue(data_dict, output_path, minV, maxV, xAxisLabel, yAxisLabel, cMapLabel, figSizeX=10, figSizeY=4, format="png", actualResGlobal = 0, ratioSignal=0):
	"""
	Create what is refered to as a p-value plot in the paper from which the 
 	median resolution is derived for tomograms and tilt-series.
	"""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import seaborn as sns
  
	# Create 2d array from heatmap
	if len(list(data_dict.keys())) > 1:
		first_key = next(iter(data_dict))
		if isinstance(first_key, str):
			sorted_keys = sorted(np.array(list(data_dict.keys())).astype(np.float32))
			sorted_values = [data_dict[str(key)] for key in sorted_keys]
		else:
			sorted_keys = sorted(data_dict.keys())
			sorted_values = [data_dict[key] for key in sorted_keys]
		sorted_keys = np.round(np.array(sorted_keys), 1)
	else:
		sorted_keys = sorted(data_dict.keys())
		sorted_values = [data_dict[key] for key in sorted_keys]

	
	heatmap_data = np.array(sorted_values)

	plt.figure(figsize=(figSizeX, figSizeY))
	plt.rcParams['font.size'] = 16
	ax = sns.heatmap(
		heatmap_data, 
		cmap="RdBu_r", 
		vmin=minV, 
		vmax=maxV, 
		xticklabels=True, 
		yticklabels=sorted_keys
	)

	ax.set_xlabel(xAxisLabel)
	ax.set_ylabel(yAxisLabel)

	cbar = ax.collections[0].colorbar  
	num_colorbar_ticks = int(figSizeY * 1.5)  
	cbar_ticks = np.linspace(minV, maxV, num_colorbar_ticks) 
	cbar.set_ticks(cbar_ticks) 
	cbar.set_label(cMapLabel) 
	cbar.set_label(cMapLabel, size=16)
	ax.collections[0].colorbar.ax.invert_yaxis()
 
	#x-axis ticks
	num_xticks = min(10, heatmap_data.shape[1]) 
	x_ticks_positions = np.linspace(0, heatmap_data.shape[1] - 1, num_xticks)
	ax.set_xticks(x_ticks_positions)
	ax.set_xticklabels([f"{int(tick):d}" for tick in x_ticks_positions])
	ax.tick_params(axis='x', labelrotation=0)

	#y-axis ticks
	num_yticks = min(10, heatmap_data.shape[0]) 
	y_ticks_positions = np.linspace(0, heatmap_data.shape[0] - 1, num_yticks)
	ax.set_yticks(y_ticks_positions)
	ax.set_yticklabels([sorted_keys[int(tick)] for tick in y_ticks_positions])

	# if (actualResGlobal != 0) and (ratioSignal != 0):
	plt.title("signal ratio " + str(ratioSignal) + "\nmedian resolution within signal " + str(actualResGlobal))
	plt.tight_layout()
	if format == "svg":
		plt.rcParams["svg.fonttype"] = "none"
		plt.savefig(output_path + ".svg", format="svg")
	elif format == "pdf":
		plt.rcParams["svg.fonttype"] = "none"
		plt.savefig(output_path + ".pdf", format="pdf")
	else:
		plt.savefig(output_path + ".png")
	plt.close()



def getWindowsEmpirical(input_values, dim):
	"""
	Includes the shell-dependent window radii for 2D and 3D measurements, calculated from our simulations.
	"""
    
	from scipy.interpolate import interp1d
	if dim == 2:
		#2DnoBorders_4udf_phaseRand_thresh0.95_molThres0.5_hypTan0.05-falloff1.5.json --- with fsc 0.2
		res = [0.1, 0.10373737373737374, 0.10747474747474747, 0.11121212121212122, 0.11494949494949495, 0.11868686868686869, 0.12242424242424244, 0.12616161616161617, 0.1298989898989899, 0.13363636363636364, 0.13737373737373737, 0.14111111111111113, 0.14484848484848484, 0.1485858585858586, 0.15232323232323233, 0.15606060606060607, 0.1597979797979798, 0.16353535353535353, 0.16727272727272727, 0.171010101010101, 0.17474747474747476, 0.17848484848484847, 0.18222222222222223, 0.18595959595959596, 0.1896969696969697, 0.19343434343434343, 0.19717171717171716, 0.20090909090909093, 0.20464646464646463, 0.2083838383838384, 0.21212121212121213, 0.21585858585858586, 0.2195959595959596, 0.22333333333333333, 0.22707070707070706, 0.2308080808080808, 0.23454545454545453, 0.2382828282828283, 0.24202020202020202, 0.24575757575757576, 0.2494949494949495, 0.25323232323232325, 0.25696969696969696, 0.26070707070707066, 0.2644444444444445, 0.2681818181818182, 0.2719191919191919, 0.27565656565656566, 0.2793939393939394, 0.2831313131313131, 0.28686868686868683, 0.2906060606060606, 0.29434343434343435, 0.29808080808080806, 0.3018181818181818, 0.3055555555555556, 0.3092929292929293, 0.313030303030303, 0.31676767676767675, 0.3205050505050505, 0.3242424242424242, 0.327979797979798, 0.33171717171717174, 0.33545454545454545, 0.33919191919191916, 0.3429292929292929, 0.3466666666666667, 0.3504040404040404, 0.3541414141414141, 0.3578787878787879, 0.3616161616161616, 0.3653535353535353, 0.369090909090909, 0.37282828282828284, 0.37656565656565655, 0.38030303030303025, 0.38404040404040407, 0.3877777777777778, 0.3915151515151515, 0.3952525252525252, 0.398989898989899, 0.4027272727272727, 0.4064646464646464, 0.41020202020202023, 0.41393939393939394, 0.41767676767676765, 0.42141414141414135, 0.42515151515151517, 0.4288888888888889, 0.4326262626262626, 0.4363636363636364, 0.4401010101010101, 0.4438383838383838, 0.4475757575757575, 0.45131313131313133, 0.45505050505050504, 0.45878787878787874, 0.46252525252525256, 0.46626262626262627, 0.47]
		windows = [49.36361869513791, 48.101260190452166, 46.86436042293574, 45.652919392588636, 44.46693709941084, 43.30641354340237, 42.17134872456322, 41.06174264289338, 39.97759529839286, 38.918906691061665, 37.88567682089978, 36.877905687907216, 35.895593292083966, 34.93873963343003, 34.00734471194542, 33.10140852763013, 32.22093108048415, 31.365912370507488, 30.536352397700146, 29.732251162062123, 28.953608663593418, 28.200424902294028, 27.472699878163954, 26.7704335912032, 26.093626041411767, 25.045565672854316, 24.398520370005397, 23.81035338175588, 23.270603830696924, 22.76867693319509, 22.289111082809953, 21.81817466072031, 21.361007203926064, 20.918186988054018, 20.486738822692256, 20.074835893503465, 19.697334857160612, 19.35801431720891, 19.0567630880189, 18.794672840227026, 18.562181792575036, 18.333626980491182, 18.09547602764037, 17.85631235059332, 17.621778528257025, 17.387999403843313, 17.15725532789141, 16.929174803548424, 16.702091760237305, 16.46919359395626, 16.2277514636398, 15.98378734028979, 15.750907875438257, 15.524943890633782, 15.297581550630724, 15.069876673755571, 14.84968026435418, 14.644798630736183, 14.458253889087974, 14.29468246400815, 14.158327090393335, 14.046873235031862, 13.951641169273067, 13.861745772148343, 13.779821423018864, 13.71046984654102, 13.656155260887243, 13.617206251565866, 13.593255117623823, 13.579620305949607, 13.576334502320877, 13.585110757334864, 13.604955215952494, 13.631404968211983, 13.663444789388635, 13.644337010640735, 13.60757937561159, 13.572765637206391, 13.539895795425144, 13.508969850267846, 13.479987801734499, 13.4529496498251, 13.427855394539652, 13.404705035878152, 13.383498573840601, 13.364236008427003, 13.346917339637352, 13.33154256747165, 13.3181116919299, 13.306624713012098, 13.297081630718246, 13.289482445048344, 13.283827156002392, 13.280115763580389, 13.278348267782336, 13.278524668608233, 13.280644966058079, 13.284709160131873, 13.29071725082962, 13.298669238151314]

	if dim == 3:
		#3D_4udf_phaseRand_thresh0.95_molThres0.3_hypTan0.05-falloff1.5
		res = [0.1, 0.10373737373737374, 0.10747474747474747, 0.11121212121212122, 0.11494949494949495, 0.11868686868686869, 0.12242424242424244, 0.12616161616161617, 0.1298989898989899, 0.13363636363636364, 0.13737373737373737, 0.14111111111111113, 0.14484848484848484, 0.1485858585858586, 0.15232323232323233, 0.15606060606060607, 0.1597979797979798, 0.16353535353535353, 0.16727272727272727, 0.171010101010101, 0.17474747474747476, 0.17848484848484847, 0.18222222222222223, 0.18595959595959596, 0.1896969696969697, 0.19343434343434343, 0.19717171717171716, 0.20090909090909093, 0.20464646464646463, 0.2083838383838384, 0.21212121212121213, 0.21585858585858586, 0.2195959595959596, 0.22333333333333333, 0.22707070707070706, 0.2308080808080808, 0.23454545454545453, 0.2382828282828283, 0.24202020202020202, 0.24575757575757576, 0.2494949494949495, 0.25323232323232325, 0.25696969696969696, 0.26070707070707066, 0.2644444444444445, 0.2681818181818182, 0.2719191919191919, 0.27565656565656566, 0.2793939393939394, 0.2831313131313131, 0.28686868686868683, 0.2906060606060606, 0.29434343434343435, 0.29808080808080806, 0.3018181818181818, 0.3055555555555556, 0.3092929292929293, 0.313030303030303, 0.31676767676767675, 0.3205050505050505, 0.3242424242424242, 0.327979797979798, 0.33171717171717174, 0.33545454545454545, 0.33919191919191916, 0.3429292929292929, 0.3466666666666667, 0.3504040404040404, 0.3541414141414141, 0.3578787878787879, 0.3616161616161616, 0.3653535353535353, 0.369090909090909, 0.37282828282828284, 0.37656565656565655, 0.38030303030303025, 0.38404040404040407, 0.3877777777777778, 0.3915151515151515, 0.3952525252525252, 0.398989898989899, 0.4027272727272727, 0.4064646464646464, 0.41020202020202023, 0.41393939393939394, 0.41767676767676765, 0.42141414141414135, 0.42515151515151517, 0.4288888888888889, 0.4326262626262626, 0.4363636363636364, 0.4401010101010101, 0.4438383838383838, 0.4475757575757575, 0.45131313131313133, 0.45505050505050504, 0.45878787878787874, 0.46252525252525256, 0.46626262626262627, 0.47]
		windows = [10.427159025265857, 10.224550648987888, 10.024724105932263, 9.827679396098981, 9.633416519488042, 9.441935476099447, 9.253236265933197, 9.067318888989288, 8.884183345267724, 8.703829634768502, 8.526257757491624, 8.35146771343709, 8.1794595026049, 8.010233124995052, 7.843788580607548, 7.680125869442387, 7.51924499149957, 7.3611459467790965, 7.205828735280966, 7.053293357005179, 6.9035398119517355, 6.756568100120637, 6.612378221511879, 6.4709701761254665, 6.332343963961397, 6.196499585019671, 6.063437039300288, 5.933156326803249, 5.805657447528553, 5.6809404014762, 5.559005188646191, 5.439851809038525, 5.323480262653203, 5.209890549490225, 5.09908266954959, 4.991056622831297, 4.885812409335349, 4.783350029061744, 4.683669482010483, 4.586770768181565, 4.49265388757499, 4.401318840190759, 4.312765626029701, 4.2443780015069645, 4.179415873372608, 4.11804602113812, 4.060340891193263, 4.006316016955979, 3.955493167471093, 3.9073774135683905, 3.8615726095125815, 3.8177800341188, 3.7757507604214515, 3.735270064878486, 3.6961391384713274, 3.6581785320472964, 3.621257850052169, 3.585270636114754, 3.5356471752345815, 3.4872090136596845, 3.439956151389373, 3.393888588423646, 3.3490063247625046, 3.3053093604059476, 3.2627976953539757, 3.2214713296065884, 3.181330263163787, 3.14237449602557, 3.1046040281919387, 3.068018859662892, 3.0326189904384298, 2.9984044205185527, 2.965375149903261, 2.9335311785925544, 2.902872506586433, 2.873399133884896, 2.845111060487944, 2.8180082863955773, 2.792090811607795, 2.767358636124599, 2.7438117599459875, 2.72145018307196, 2.7002739055025184, 2.6802829272376627, 2.6614772482773903, 2.643856868621704, 2.6274217882706026, 2.6121720072240855, 2.5981075254821535, 2.5852283430448075, 2.573534459912045, 2.563025876083869, 2.5537025915602776, 2.5455646063412702, 2.538611920426849, 2.532844533817012, 2.5282624465117607, 2.524865658511094, 2.522654169815012, 2.5216279804235153]



	interp_func = interp1d(res, windows, kind='linear', fill_value="extrapolate")
	output = np.array(interp_func(input_values).tolist()) 

	return output 



def correlationCoefficient(sample1, sample2):
	"""
	Calculation the correlation via Cosine similarity.
	"""
 
	FSCnominator = np.dot(sample1, sample2)
	FSCdenominator = np.linalg.norm(sample1)*np.linalg.norm(sample2)

	
	if (FSCdenominator == 0) or (np.isnan(FSCnominator)) or (np.isnan(FSCdenominator)):
		FSC = 0.0
	else:
		FSC = FSCnominator/FSCdenominator
 
	return FSC



def calculate_frequency_map(sizeMap):
	"""
	Calculation the frequency map for 2D or 3D.
	"""
	if len(sizeMap) == 3:
		freqi = np.fft.fftfreq(sizeMap[0], 1.0)
		freqj = np.fft.fftfreq(sizeMap[1], 1.0)
		freqk = np.fft.rfftfreq(sizeMap[2], 1.0)

		sizeFFT = np.array([freqi.size, freqj.size, freqk.size])
		FFT = np.zeros(sizeFFT)

		freqMapi = np.copy(FFT)
		for j in range(sizeFFT[1]):
			for k in range(sizeFFT[2]):
				freqMapi[:, j, k] = freqi * freqi

		freqMapj = np.copy(FFT)
		for i in range(sizeFFT[0]):
			for k in range(sizeFFT[2]):
				freqMapj[i, :, k] = freqj * freqj

		freqMapk = np.copy(FFT)
		for i in range(sizeFFT[0]):
			for j in range(sizeFFT[1]):
				freqMapk[i, j, :] = freqk * freqk

		frequencyMap = np.sqrt(freqMapi + freqMapj + freqMapk)

	elif len(sizeMap) == 2:
		# calc frequency for each voxel
		freqi = np.fft.fftfreq(sizeMap[0], 1.0)
		freqj = np.fft.rfftfreq(sizeMap[1], 1.0)

		sizeFFT = np.array([freqi.size, freqj.size])
		FFT = np.zeros(sizeFFT)

		freqMapi = np.copy(FFT)
		for j in range(sizeFFT[1]):
			freqMapi[:, j] = freqi * freqi

		freqMapj = np.copy(FFT)
		for i in range(sizeFFT[0]):
			freqMapj[i, :] = freqj * freqj

		frequencyMap = np.sqrt(freqMapi + freqMapj)
	return frequencyMap


	
def calculateShells(sizeVol, apix, resMax, spacingFilter, accuracy_steps):
	"""
	Calculate the resolution shells.
	"""
	freqMap = np.fft.rfftfreq(sizeVol, 1)
	freqMap = freqMap[1:] #discard zero component
	resMap = [1/i for i in freqMap]
	freqMap_apix = freqMap/apix
	nyq = 0.5
	shells = {}
	for index_i, i in enumerate(resMap):
		if (index_i%accuracy_steps != 0) & (index_i != len(resMap)-1): continue
		shell1 = (freqMap[index_i] - (nyq*spacingFilter))/apix
		shell2 = (freqMap[index_i] + (nyq*spacingFilter))/apix
		res = 1/freqMap_apix[index_i]
		if res > resMax: continue 
		shells[1/res] = ([shell1, shell2])
	return shells


def interpolate_with_zoom(input, outputShape, stepSize, lowRes):
	"""
	Map interpolation with ndimage.zoom function.
	"""
	stepArray = [(range(0, outputShape[i], stepSize[i])) for i in range(len(outputShape))]
	maxValue = [np.max(arr)+1 for arr in stepArray]
	zoom_factors = np.array(maxValue)/np.array(input.shape)

	zoomedMap = ndimage.zoom(input, zoom_factors, order=1)
	outputMap = np.zeros(outputShape)
	outputMap.fill(lowRes)
	slices = [slice(0, zoomedMap.shape[index_i]) for index_i in range(len(outputShape))]
	outputMap[tuple(slices)] = zoomedMap
	return outputMap


def interpolate_chunks(input_data, output_shape, dimension, iterate_boxSize, localResMap_size, localResMap_out_size, stepSize, chunk_shape):
	"""
	Map interpolation for large maps, interpolate grid chunk-wise
	"""
	print("Interpolate Grid chunk-wise")

	input_coords = [[] for _ in range(dimension)]
	for i in range(iterate_boxSize[0]):
		for j in range(localResMap_size[0]):
			xRange = int((i*localResMap_size[0])+j)
			if xRange >= localResMap_out_size[0]: continue      
			input_coords[0].append(xRange*stepSize[0])
	for i in range(iterate_boxSize[1]):
		for j in range(localResMap_size[1]):
			yRange = int((i*localResMap_size[1])+j)  
			if yRange >= localResMap_out_size[1]: continue       
			input_coords[1].append(yRange*stepSize[1])
	if dimension == 3:
		for i in range(iterate_boxSize[2]):
			for j in range(localResMap_size[2]):
				zRange = int((i*localResMap_size[2])+j)     
				if zRange >= localResMap_out_size[2]: continue    
				input_coords[2].append(zRange*stepSize[2])      
	
	interpolator = RegularGridInterpolator(input_coords, input_data, method='linear', bounds_error=False)

	output = np.empty(output_shape)
	chunk_size = np.ceil(np.divide(output_shape, chunk_shape)).astype(int)

	for i in range(chunk_size[0]):
		for j in range(chunk_size[1]):
			for k in range(chunk_size[2]):
				# Define chunk boundaries
				start_x = i * chunk_shape[0]
				end_x = min((i + 1) * chunk_shape[0], output_shape[0])
				start_y = j * chunk_shape[1]
				end_y = min((j + 1) * chunk_shape[1], output_shape[1])
				start_z = k * chunk_shape[2]
				end_z = min((k + 1) * chunk_shape[2], output_shape[2])

				# Generate meshgrid for the chunk
				x, y, z = np.meshgrid(np.linspace(start_x, end_x - 1, end_x - start_x),
									   np.linspace(start_y, end_y - 1, end_y - start_y),
									   np.linspace(start_z, end_z - 1, end_z - start_z),
									   indexing='ij')

				# Interpolate the chunk
				interpolated_chunk = interpolator((x,y,z))

				# Assign the interpolated chunk to the output
				output[start_x:end_x, start_y:end_y, start_z:end_z] = interpolated_chunk

	return output

def interpolate2d_Clough(zipped, z):
	"""
	Interpolation function for 2D data.
	"""
	from scipy.interpolate import CloughTocher2DInterpolator
	interp = CloughTocher2DInterpolator(zipped, z, rescale=True)
	return interp


def calculateEfficientBoxSize(sizeMap_input, boxValue, maxWindow_half, runOnGpu, dimension_windows, collapseWindow_i = False):
	"""
	Calculate box size in order to add zero padding to map to make FFT calculations as efficient as possible.
	"""		
  
	def find_next_good_size(n, dim):
		if dim == 2:
			good_sizes = [300, 320, 324, 336, 384, 400, 432, 448, 450, 512, 576, 640, 648, 672, 720, 768, 784, 810, 864, 882, 1024, 1152, 1280, 1296, 1344, 1440, 1568, 1620, 1728, 1792, 2000, 2048, 2160, 2592, 2744, 3456, 4096]
		if dim== 3:
			good_sizes = [160, 180, 192, 200, 216, 224, 240, 256, 270, 288, 300, 320, 324, 336, 384, 400, 432, 448, 450, 512, 576, 640, 648, 672, 720, 768, 784, 810, 864, 882, 1024, 1152, 1280, 1296, 1344, 1440, 1568, 1620, 1728, 1792, 2000, 2048, 2160, 2592, 2744, 3456, 4096]
		for size in good_sizes:
			if size >= n:
				return size
		# Fall back to next power of 2 for very large sizes
		return 2**int(np.ceil(np.log2(n)))

	boxSize = np.zeros(len(sizeMap_input))
	corrected_box_size = np.zeros(len(sizeMap_input))

	if collapseWindow_i:
		boxSize[0] = sizeMap_input[0]
		corrected_box_size[0] = sizeMap_input[0]
		for i in range(1,3):
			boxSize[i] = find_next_good_size(sizeMap_input[i]+(2*maxWindow_half[i]), 2)
			corrected_box_size[i] = boxSize[i] - (maxWindow_half[i]*2)
	
		return boxSize.astype(np.int32), corrected_box_size.astype(np.int32)

	#set boxSize to an efficient size for pyfftw
	if len(sizeMap_input) == 3:
		for i in range(3):
			boxSize[i] = find_next_good_size(sizeMap_input[i]+(2*maxWindow_half[i]), 3)
		corrected_box_size = np.array([boxSize[0] - (maxWindow_half[0]*2), boxSize[1] - (maxWindow_half[1]*2), boxSize[2] - (maxWindow_half[2]*2)])
  
	if len(sizeMap_input) == 2:
		for i in range(2):
			boxSize[i] = find_next_good_size(sizeMap_input[i]+(2*maxWindow_half[i]), 2)
		corrected_box_size = np.array([boxSize[0] - (maxWindow_half[0]*2), boxSize[1] - (maxWindow_half[1]*2)])

	return boxSize.astype(np.int32), corrected_box_size.astype(np.int32)


def calcResIndex(qVals_FDR, p_cutoff, test2):
	"""
	CPU functionality to calculate q-value cutoff.
	"""		
	res_index = -1
	testing = True
	for x in range(len(qVals_FDR)):
		if qVals_FDR[x] <= p_cutoff and testing:
			res_index = x
			testing = True
		else:
			if not test2:
				break
			if test2 and testing:
				res_index = x
			if test2 and not testing:
				res_index -= 1
				break
			testing = False

	return res_index

def hypTan_cuda(apix, fftmap, res_obj_inv, cuda_distance, low_freq, high_freq, falloff, gpu_id, analyze=False, cuda_bandpassIn=[], separateXSlices = False):
	"""
	Hyperbolic tangent bandpass filter for GPU.
	"""		

	from numba import cuda
	cuda.select_device(gpu_id)
	stream = cuda.stream()
 
	# 1. general
	if len(fftmap.shape) == 3:
		threadsperblock = (8, 8, 8)
		blockspergrid_x = (fftmap.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
		blockspergrid_y = (fftmap.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
		blockspergrid_z = (fftmap.shape[2] + threadsperblock[2] - 1) // threadsperblock[2]
		blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
	if (len(fftmap.shape) == 2):
		threadsperblock = (8, 8)
		blockspergrid_x = (fftmap.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
		blockspergrid_y = (fftmap.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
		blockspergrid = (blockspergrid_x, blockspergrid_y)		
	if (len(fftmap.shape) == 3) and separateXSlices:
		threadsperblock = (8, 8)
		blockspergrid_x = (fftmap.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
		blockspergrid_y = (fftmap.shape[2] + threadsperblock[1] - 1) // threadsperblock[1]
		blockspergrid = (blockspergrid_x, blockspergrid_y)	  
  
	# 2. if bandpass filtered already calculated, do calculations
	if len(cuda_bandpassIn) > 0:
		# print(fftmap)
		# Configure the GPU grid and block dimensions
		product = np.zeros_like(fftmap)
		cuda_product = cuda.to_device(product, stream=stream)
		cuda_fftmap = cuda.to_device(fftmap, stream=stream)
		# cuda_bandpassIn = cuda.to_device(bandpassIn, stream=stream)
		cuda_shapeMap = cuda.to_device(fftmap.shape, stream=stream)




		@cuda.jit
		def cuda_multiply3D(cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap):
			i, j, k = cuda.grid(3)
			if i < cuda_shapeMap[0] and j < cuda_shapeMap[1] and k < cuda_shapeMap[2]:
				cuda_product[i, j, k] = cuda_fftmap[i,j,k] * cuda_bandpassIn[i,j,k]
	
		@cuda.jit
		def cuda_multiply2D(cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap):
			i, j = cuda.grid(2)
			if i < cuda_shapeMap[0] and j < cuda_shapeMap[1]:
				cuda_product[i, j] = cuda_fftmap[i,j] * cuda_bandpassIn[i,j]
	 
		@cuda.jit
		def cuda_multiply3D_divByX(cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap):
			i, j = cuda.grid(2)
			if i < cuda_shapeMap[1] and j < cuda_shapeMap[2]:
				for x in range(cuda_shapeMap[0]):
					cuda_product[x, i, j] = cuda_fftmap[x,i,j] * cuda_bandpassIn[i,j]
	 

		if separateXSlices and (len(fftmap.shape) == 3):
			cuda_multiply3D_divByX[blockspergrid, threadsperblock](cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap)
		else:
			if len(fftmap.shape) == 3:
				cuda_multiply3D[blockspergrid, threadsperblock](cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap)  
	
			if len(fftmap.shape) == 2:
				cuda_multiply2D[blockspergrid, threadsperblock](cuda_product, cuda_fftmap, cuda_bandpassIn, cuda_shapeMap)
   
		stream.synchronize()
		product = cuda_product.copy_to_host(stream=stream)
		# cuda.current_context().reset()

		if separateXSlices and (len(fftmap.shape) == 3):
			returnThis = []
			for x in range(product.shape[0]):
				returnThis.append(np.copy(res_obj_inv(product[x])))
		else:
			returnThis = np.copy(res_obj_inv(product))

		del cuda_product 
		del cuda_fftmap
		del cuda_shapeMap
		cuda.current_context().memory_manager.deallocations.clear()



		return returnThis

	# 3. create the bandpass filter
	@cuda.jit
	def cuda_bandpass_filter3D(cuda_bandpass, cuda_distance, low_freq, high_freq, low_fall_off, high_fall_off, cuda_shapeMap):
		i, j, k = cuda.grid(3)
		if i < cuda_shapeMap[0] and j < cuda_shapeMap[1] and k < cuda_shapeMap[2]:
			cuda_bandpass[i, j, k] = 0.5 * (
				np.tanh(np.pi * (cuda_distance[i, j, k] + low_freq) / (low_fall_off * (low_freq - high_freq)))
				- np.tanh(np.pi * (cuda_distance[i, j, k] - low_freq) / (low_fall_off * (low_freq - high_freq)))
				- np.tanh(np.pi * (cuda_distance[i, j, k] + high_freq) / (high_fall_off * (low_freq - high_freq)))
				+ np.tanh(np.pi * (cuda_distance[i, j, k] - high_freq) / (high_fall_off * (low_freq - high_freq)))
			)

	@cuda.jit
	def cuda_bandpass_filter2D(cuda_bandpass, cuda_distance, low_freq, high_freq, low_fall_off, high_fall_off, cuda_shapeMap):
		i, j = cuda.grid(2)
		if i < cuda_shapeMap[0] and j < cuda_shapeMap[1]:
			cuda_bandpass[i, j] = 0.5 * (
				np.tanh(np.pi * (cuda_distance[i, j] + low_freq) / (low_fall_off * (low_freq - high_freq)))
				- np.tanh(np.pi * (cuda_distance[i, j] - low_freq) / (low_fall_off * (low_freq - high_freq)))
				- np.tanh(np.pi * (cuda_distance[i, j] + high_freq) / (high_fall_off * (low_freq - high_freq)))
				+ np.tanh(np.pi * (cuda_distance[i, j] - high_freq) / (high_fall_off * (low_freq - high_freq)))
			)
   
	high_fall_off = falloff
	low_fall_off = falloff

	bandpass_filter = np.zeros(fftmap.shape)
	# cuda_distance = cuda.to_device(distance, stream=stream)
	cuda_bandpass = cuda.to_device(bandpass_filter, stream=stream)
	cuda_shapeMap = cuda.to_device(fftmap.shape, stream=stream)

	if len(fftmap.shape) == 3:
		cuda_bandpass_filter3D[blockspergrid, threadsperblock](cuda_bandpass, cuda_distance, low_freq, high_freq, low_fall_off, high_fall_off, cuda_shapeMap)
  
	if len(fftmap.shape) == 2:
		cuda_bandpass_filter2D[blockspergrid, threadsperblock](cuda_bandpass, cuda_distance, low_freq, high_freq, low_fall_off, high_fall_off, cuda_shapeMap)
   
	stream.synchronize()

 
	@cuda.jit
	def cuda_bandpass_normalize3D(cuda_bandpass, max_val):
		i, j, k = cuda.grid(3)
		if i < cuda_bandpass.shape[0] and j < cuda_bandpass.shape[1] and k < cuda_bandpass.shape[2]:
			cuda_bandpass[i, j, k] =  (cuda_bandpass[i, j, k] / max_val)

	@cuda.jit
	def cuda_bandpass_normalize2D(cuda_bandpass, max_val):
		i, j = cuda.grid(2)
		if i < cuda_bandpass.shape[0] and j < cuda_bandpass.shape[1]:
			cuda_bandpass[i, j] =  (cuda_bandpass[i, j] / max_val)


	max_val = float(np.max(cuda_bandpass.copy_to_host()))
	if len(fftmap.shape) == 3:
		cuda_bandpass_normalize3D[blockspergrid, threadsperblock](cuda_bandpass, max_val)

	if len(fftmap.shape) == 2:
		cuda_bandpass_normalize2D[blockspergrid, threadsperblock](cuda_bandpass, max_val)

	stream.synchronize()

 
	if analyze: 
		return cuda_bandpass

	resulta = np.multiply(fftmap,cuda_bandpass.copy_to_host(stream=stream))
	x = np.float32(np.copy(res_obj_inv(resulta)))

	return x

def hypTan(apix, fftmap, res_obj_inv, distance, high_freq, low_freq, falloff, runOnGPU, analyze=False, bandpassIn=[], separateXSlices = False):
	"""
	Hyperbolic tangent bandpass filter for CPU.
	"""		
	if len(bandpassIn) > 0:
	
		if separateXSlices:
			returnThis = []
			for x in range(fftmap.shape[0]):
				output = np.multiply(fftmap[x], bandpassIn)
				returnThis.append(np.copy(res_obj_inv(output)))
		else:
			output = np.multiply(fftmap,bandpassIn)
			returnThis = np.copy(res_obj_inv(output))
		return returnThis
 
	high_fall_off = falloff
	low_fall_off = falloff

 
	bandpass_filter = 0.5*(np.tanh(np.pi*(distance + low_freq)/(low_fall_off*(low_freq - high_freq))) 
				- np.tanh(np.pi*(distance - low_freq)/(low_fall_off*(low_freq - high_freq)))
				- np.tanh(np.pi*(distance + high_freq)/(high_fall_off*(low_freq - high_freq))) 
				+ np.tanh(np.pi*(distance - high_freq)/(high_fall_off*(low_freq - high_freq))))
	
	bandpass_filter/=np.max(bandpass_filter)
	if analyze: 
		return bandpass_filter

	x = np.float32(np.copy(res_obj_inv(np.multiply(fftmap,bandpass_filter))))
	return x


def runLocal_cuda(corrBoxSize, maxWindow_half, window_size, window_size_i, paddedHalfMap1, paddedHalfMap2, 
							stepSize, permuted_map, bool_array, start_gpu, gpu_id, result_array_gpu, dimsMin, dimsMax):
	"""
	Local correlation measurements for GPU - optimized.
	"""		
	from numba import cuda
	import numpy as np
	
	cuda.select_device(gpu_id)
	stream = cuda.stream()
	
	# Pre-compute valid indices to avoid modulo operations
	dim = len(paddedHalfMap1.shape)
	
	# Pre-process bool_array to get only valid indices
	bool_indices = np.argwhere(bool_array == 1).astype(np.int32)
	num_valid_points = len(bool_indices)
	
	# Convert tuple parameters to numpy arrays for CUDA compatibility
	corrBoxSize_arr = np.array(corrBoxSize, dtype=np.int32)
	maxWindow_half_arr = np.array(maxWindow_half, dtype=np.int32)
	stepSize_arr = np.array(stepSize, dtype=np.int32)
	
	# Optimize thread block configuration based on actual workload
	if dim == 3:
		# Use 1D thread blocks for better coalescing when processing windows
		threadsperblock = 512
		# Calculate actual number of positions to process
		num_positions = (corrBoxSize[0] // stepSize[0]) * (corrBoxSize[1] // stepSize[1]) * (corrBoxSize[2] // stepSize[2])
		blockspergrid = (num_positions + threadsperblock - 1) // threadsperblock
	else:
		threadsperblock = 512
		num_positions = (corrBoxSize[0] // stepSize[0]) * (corrBoxSize[1] // stepSize[1])
		blockspergrid = (num_positions + threadsperblock - 1) // threadsperblock
	
	# Transfer data to GPU
	cuda_resultArray = cuda.to_device(result_array_gpu, stream=stream)
	cuda_paddedHalfMap1 = cuda.to_device(paddedHalfMap1, stream=stream)
	cuda_paddedHalfMap2 = cuda.to_device(paddedHalfMap2, stream=stream)
	cuda_permuted_map = cuda.to_device(permuted_map, stream=stream)
	cuda_bool_indices = cuda.to_device(bool_indices, stream=stream)
	cuda_corrBoxSize = cuda.to_device(corrBoxSize_arr, stream=stream)
	cuda_maxWindow_half = cuda.to_device(maxWindow_half_arr, stream=stream)
	cuda_stepSize = cuda.to_device(stepSize_arr, stream=stream)
	
	# Pre-calculate constants
	permutedMapLen = len(permuted_map)
	
	@cuda.jit
	def run_cuda_2d(cuda_resultArray, window_size, cuda_paddedHalfMap1, cuda_paddedHalfMap2, 
							  cuda_bool_indices, num_valid_points, cuda_maxWindow_half, cuda_stepSize, 
							  cuda_permuted_map, permutedMapLen, cuda_corrBoxSize):
		
		tid = cuda.grid(1)
		
		# Calculate dimensions of result array
		result_dim0 = cuda_corrBoxSize[0] // cuda_stepSize[0]
		result_dim1 = cuda_corrBoxSize[1] // cuda_stepSize[1]
		total_positions = result_dim0 * result_dim1
		
		if tid >= total_positions:
			return
		
		# Convert linear index to 2D position in result array
		iInd = tid // result_dim1
		jInd = tid % result_dim1
		
		# Calculate actual position in padded map
		i = cuda_maxWindow_half[0] + iInd * cuda_stepSize[0]
		j = cuda_maxWindow_half[1] + jInd * cuda_stepSize[1]
		
		# Calculate FSC using only valid points
		FSCnominator = 0.0
		sum1 = 0.0
		sum2 = 0.0
		
		# Process all valid points directly
		for idx in range(num_valid_points):
			l = cuda_bool_indices[idx, 0]
			m = cuda_bool_indices[idx, 1]
			
			# Calculate actual indices in the window
			idx1 = i - window_size + l
			idx2 = j - window_size + m
			
			val1 = cuda_paddedHalfMap1[idx1, idx2]
			val2 = cuda_paddedHalfMap2[idx1, idx2]
			
			FSCnominator += val1 * val2
			sum1 += val1 * val1
			sum2 += val2 * val2
		
		# Calculate FSC
		FSCdenominator = math.sqrt(sum1) * math.sqrt(sum2)
		FSC = 0.0 if FSCdenominator == 0 else FSCnominator / FSCdenominator
		
		# Count permutations
		countP = 0
		for idx in range(permutedMapLen):
			if cuda_permuted_map[idx] > FSC:
				countP += 1
		
		# Store result
		cuda_resultArray[iInd, jInd] = countP / float(permutedMapLen)
	
	@cuda.jit
	def run_cuda_3d(cuda_resultArray, window_size, window_size_i, cuda_paddedHalfMap1, cuda_paddedHalfMap2, 
							  cuda_bool_indices, num_valid_points, cuda_maxWindow_half, cuda_stepSize, 
							  cuda_permuted_map, permutedMapLen, cuda_corrBoxSize):
		
		tid = cuda.grid(1)
		
		# Calculate dimensions of result array
		result_dim0 = cuda_corrBoxSize[0] // cuda_stepSize[0]
		result_dim1 = cuda_corrBoxSize[1] // cuda_stepSize[1]
		result_dim2 = cuda_corrBoxSize[2] // cuda_stepSize[2]
		total_positions = result_dim0 * result_dim1 * result_dim2
		
		if tid >= total_positions:
			return
		
		# Convert linear index to 3D position
		temp = tid
		kInd = temp % result_dim2
		temp = temp // result_dim2
		jInd = temp % result_dim1
		iInd = temp // result_dim1
		
		# Calculate actual position in padded map
		i = cuda_maxWindow_half[0] + iInd * cuda_stepSize[0]
		j = cuda_maxWindow_half[1] + jInd * cuda_stepSize[1]
		k = cuda_maxWindow_half[2] + kInd * cuda_stepSize[2]
		
		# Calculate FSC using only valid points
		FSCnominator = 0.0
		sum1 = 0.0
		sum2 = 0.0
		
		# Process all valid points directly
		for idx in range(num_valid_points):
			l = cuda_bool_indices[idx, 0]
			m = cuda_bool_indices[idx, 1]
			n = cuda_bool_indices[idx, 2]
			
			# Calculate actual indices
			idx1 = i - window_size_i + l
			idx2 = j - window_size + m
			idx3 = k - window_size + n
			
			val1 = cuda_paddedHalfMap1[idx1, idx2, idx3]
			val2 = cuda_paddedHalfMap2[idx1, idx2, idx3]
			
			FSCnominator += val1 * val2
			sum1 += val1 * val1
			sum2 += val2 * val2
		
		# Calculate FSC
		FSCdenominator = math.sqrt(sum1) * math.sqrt(sum2)
		FSC = 0.0 if FSCdenominator == 0 else FSCnominator / FSCdenominator
		
		# Count permutations
		countP = 0
		for idx in range(permutedMapLen):
			if cuda_permuted_map[idx] > FSC:
				countP += 1
		
		# Store result
		cuda_resultArray[iInd, jInd, kInd] = countP / float(permutedMapLen)
	
	# Launch appropriate kernel
	if dim == 2:
		run_cuda_2d[blockspergrid, threadsperblock](
			cuda_resultArray, window_size, cuda_paddedHalfMap1, cuda_paddedHalfMap2,
			cuda_bool_indices, num_valid_points, cuda_maxWindow_half, cuda_stepSize,
			cuda_permuted_map, permutedMapLen, cuda_corrBoxSize
		)
	else:
		run_cuda_3d[blockspergrid, threadsperblock](
			cuda_resultArray, window_size, window_size_i, cuda_paddedHalfMap1, cuda_paddedHalfMap2,
			cuda_bool_indices, num_valid_points, cuda_maxWindow_half, cuda_stepSize,
			cuda_permuted_map, permutedMapLen, cuda_corrBoxSize
		)
	
	# Synchronize and copy back
	stream.synchronize()
	cuda_resultArray.copy_to_host(result_array_gpu, stream=stream)
	


def fillMap_cuda(i_pre, j_pre, k_pre, localResMap_size, localResMap_out, locResMap, p_cutoff, test2, res, lowRes, numPval, gpu_id):
	"""
	Filling map on GPU (from per-shell maps with p-values to one q-thresholded resolution map).
	"""
	import numba
	from numba import cuda
	cuda.select_device(gpu_id)
	stream = cuda.stream()
 
	locResMap = np.array(locResMap)
	res = np.array(res)

	# cp_locResMap = np.zeros_like(localResMap_out)
	cuda_localResMap_out = cuda.to_device(localResMap_out, stream=stream)
	cuda_locResMap = cuda.to_device(locResMap, stream=stream)
	# meminfo = cuda.current_context().get_memory_info()
	# print(f"Free memory: {meminfo.free / 1024**3:.2f} GB")
	# print(f"Used memory: {(meminfo.total - meminfo.free) / 1024**3:.2f} GB")
	cuda_res = cuda.to_device(res, stream=stream)

	std = 1.0
	w_edge = math.exp(-0.5 * (1.0 / std)**2)
	w_center = 1.0
	total = w_center + 2 * w_edge
	w_edge /= total
	w_center /= total
 
 
	# from numba.cuda import current_context
	# free, total = current_context().get_memory_info()
	# print(f"Free memory: {free}, Total memory: {total}")

	if len(localResMap_size) == 3:
		threadsperblock = (8, 8, 8)
		blockspergrid_x = (localResMap_size[0] + threadsperblock[0] - 1) // threadsperblock[0]
		blockspergrid_y = (localResMap_size[1] + threadsperblock[1] - 1) // threadsperblock[1]
		blockspergrid_z = (localResMap_size[2] + threadsperblock[2] - 1) // threadsperblock[2]
		blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
	if len(localResMap_size) == 2:
		threadsperblock = (8, 8)
		blockspergrid_x = (localResMap_size[0] + threadsperblock[0] - 1) // threadsperblock[0]
		blockspergrid_y = (localResMap_size[1] + threadsperblock[1] - 1) // threadsperblock[1]
		blockspergrid = (blockspergrid_x, blockspergrid_y)
  
	Hn = math.log(numPval) + 0.5772 + 0.5/numPval - 1.0/(12*numPval**2) + 1.0/(120*numPval**4)   
 
 
	@cuda.jit
	def median_cuda(arr, n):
		temp = cuda.local.array(50, numba.float32)
		for i in range(n):
			temp[i] = arr[i]
		for i in range(1, n):
			key = temp[i]
			j = i - 1
			while j >= 0 and temp[j] > key:
				temp[j + 1] = temp[j]
				j -= 1
			temp[j + 1] = key
		if n % 2 == 1:
			return temp[n // 2]
		else:
			return (temp[(n // 2) - 1] + temp[n // 2]) / 2.0
 
	@cuda.jit
	def run_cuda_FDR2d(i_pre, j_pre, k_pre, localResMap_size, cuda_localResMap_out, cuda_locResMap, p_cutoff, cuda_res, lowRes, numPval, Hn, w_edge, w_center):
		i, j = cuda.grid(2)
		# localResMap_size = localResMap_size.shape
		iInd = int((i_pre*localResMap_size[0])+i)
		jInd = int((j_pre*localResMap_size[1])+j)
	
		pVals = cuda.local.array(50, numba.float32)
		if (iInd < cuda_localResMap_out.shape[0]) and (jInd < cuda_localResMap_out.shape[1]) and (i < localResMap_size[0]) and (j < localResMap_size[1]):
			for m in range(numPval):
				pVals[m] = (cuda_locResMap[m][i][j])

			#FDR correction
			#argsort manual implementation
			indices = cuda.local.array(50, numba.int16)
			for x in range(50):
				indices[x] = x
			for x in range(numPval):
				for y in range(x + 1, numPval):
					if pVals[indices[x]] > pVals[indices[y]]:
						temp = indices[x]
						indices[x] = indices[y]
						indices[y] = temp
			
			pSort = cuda.local.array(50, numba.float32)
			for x in range(numPval):
				pSort[x] = pVals[indices[x]]
				
			#adjust pVal
			prevPVal = 1.0
			for x in range(numPval-1, -1, -1):
				if prevPVal < (pSort[x]*(numPval/(x+1.0))*Hn):
					pVals[x] = prevPVal
				else:
					pVals[x] = (pSort[x]*(numPval/(x+1.0))*Hn)
				prevPVal = pVals[x]  
			
			#resort array
			indices2 = cuda.local.array(50, numba.int16)
			for x in range(50):
				indices2[x] = x
			for x in range(numPval):
				for y in range(x + 1, numPval):
					if indices[indices2[x]] > indices[indices2[y]]:
						temp = indices2[x]
						indices2[x] = indices2[y]
						indices2[y] = temp
						
			for x in range(numPval):
				pSort[x] = pVals[indices2[x]]        
		
			res_index = -1
			testing = True
			for x in range(numPval):
				if pSort[x] <= p_cutoff and testing:
					res_index = x
					testing = True
				else:
					if not test2:
						break
					if test2 and testing:
						res_index = x
					if test2 and not testing:
						res_index -= 1
						break
					testing = False

			
			if res_index < 0:
				cuda_localResMap_out[iInd][jInd]= lowRes
			else:
				resolution = int((1/cuda_res[res_index])*10)/10 #round
				cuda_localResMap_out[iInd][jInd] = resolution 
 
 
 
	@cuda.jit
	def run_cuda_FDR3d(i_pre, j_pre, k_pre, localResMap_size, cuda_localResMap_out, cuda_locResMap, p_cutoff, cuda_res, lowRes, numPval, Hn, w_edge, w_center):
		i, j, k = cuda.grid(3)
		# localResMap_size = localResMap_size.shape
		iInd = int((i_pre*localResMap_size[0])+i)
		jInd = int((j_pre*localResMap_size[1])+j)
		kInd = int((k_pre*localResMap_size[2])+k)
	
		pVals = cuda.local.array(50, numba.float32)
		if (iInd < cuda_localResMap_out.shape[0]) and (jInd < cuda_localResMap_out.shape[1]) and (kInd < cuda_localResMap_out.shape[2]) and (i < localResMap_size[0]) and (j < localResMap_size[1]) and (k < localResMap_size[2]):
			for m in range(numPval):
				pVals[m] = (cuda_locResMap[m][i][j][k])

			#FDR correction
			#argsort manual implementation
			indices = cuda.local.array(50, numba.int16)
			for x in range(50):
				indices[x] = x
			for x in range(numPval):
				for y in range(x + 1, numPval):
					if pVals[indices[x]] > pVals[indices[y]]:
						temp = indices[x]
						indices[x] = indices[y]
						indices[y] = temp
			
			pSort = cuda.local.array(50, numba.float32)
			for x in range(numPval):
				pSort[x] = pVals[indices[x]]
				
			#adjust pVal
			prevPVal = 1.0
			for x in range(numPval-1, -1, -1):
				if prevPVal < (pSort[x]*(numPval/(x+1.0))*Hn):
					pVals[x] = prevPVal
				else:
					pVals[x] = (pSort[x]*(numPval/(x+1.0))*Hn)
				prevPVal = pVals[x]  
			
			#resort array
			indices2 = cuda.local.array(50, numba.int16)
			for x in range(50):
				indices2[x] = x
			for x in range(numPval):
				for y in range(x + 1, numPval):
					if indices[indices2[x]] > indices[indices2[y]]:
						temp = indices2[x]
						indices2[x] = indices2[y]
						indices2[y] = temp
						
			for x in range(numPval):
				pSort[x] = pVals[indices2[x]]        
		
			# --> pSort is the result
			res_index = -1
			for x in range(numPval):
				if pSort[x] <= p_cutoff:
					res_index = x
				else:
					break
			
			if res_index < 0:
				cuda_localResMap_out[iInd][jInd][kInd] = lowRes
			else:
				resolution = int((1/cuda_res[res_index])*100)/100 #round
				cuda_localResMap_out[iInd][jInd][kInd] = resolution
	
	
	
	if len(localResMap_size) == 3:
		run_cuda_FDR3d[blockspergrid, threadsperblock](i_pre, j_pre, k_pre, localResMap_size, cuda_localResMap_out, cuda_locResMap, p_cutoff, cuda_res, lowRes, numPval, Hn, w_edge, w_center)

	if len(localResMap_size) == 2:
		run_cuda_FDR2d[blockspergrid, threadsperblock](i_pre, j_pre, k_pre, localResMap_size, cuda_localResMap_out, cuda_locResMap, p_cutoff, cuda_res, lowRes, numPval, Hn, w_edge, w_center)
	
 
	stream.synchronize()
	cuda_localResMap_out.copy_to_host(localResMap_out, stream = stream)
	del cuda_localResMap_out
	del cuda_locResMap
	del cuda_res
	cuda.current_context().memory_manager.deallocations.clear()

 
 
	return localResMap_out



def fillMapMultiThread(queue, j, k, l, shape0Range, localResMap_size, localResMap_out, locResMap,res, p_cutoff, test2, dimension):
	"""
	Filling map on CPU, multi-threaded (from per-shell maps with p-values to one q-thresholded resolution map).
	"""
 
	for j2 in range(shape0Range[0], shape0Range[1]):
		jInd = int((j*localResMap_size[0])+j2)
		if jInd >= localResMap_out.shape[0]: continue # here we just skip the overhang due to the larger box we take to gain efficiency from the pyfftw library
		
		for k2 in range(localResMap_size[1]):
			kInd = int((k*localResMap_size[1])+k2)
			if kInd >= localResMap_out.shape[1]: continue

			if dimension == 3:
				for l2 in range(localResMap_size[2]):
					lInd = int((l*localResMap_size[2])+l2)
					if lInd >= localResMap_out.shape[2]: continue
					
					pVals = []
					# fscVals = []
					for m in range(len(locResMap)):
						pVals.append(locResMap[m][j2][k2][l2])
					qVals_FDR = pAdjust_BY(np.array(pVals))
					res_index = calcResIndex(np.copy(qVals_FDR), p_cutoff, test2)
					if res_index < 0: #only if first shell failed. 
						continue
					localResMap_out[jInd,kInd,lInd] = int((1/res[res_index])*100)/100 #np.round(1/(res[res_index]), 4)
			if dimension == 2:
				pVals = []
				for m in locResMap:
					pVals.append(m[j2][k2])
				
				qVals_FDR = pAdjust_BY(np.array(pVals))
				res_index = calcResIndex(np.copy(qVals_FDR), p_cutoff, test2)
				
				if res_index < 0: #only if first shell failed. 
					continue
				localResMap_out[jInd,kInd] = int((1/res[res_index])*100)/100
	




def fillMap(j, k, l, localResMap_size, localResMap_out, locResMap, p_cutoff, test2, res, dimension):
	"""
	Filling map on CPU, single thread (from per-shell maps with p-values to one q-thresholded resolution map).
	"""
	for j2 in range(localResMap_size[0]):
		jInd = int((j*localResMap_size[0])+j2)
		if jInd >= localResMap_out.shape[0]: continue # here we just skip the overhang due to the larger box we take to gain efficiency from the pyfftw library
		
		for k2 in range(localResMap_size[1]):
			kInd = int((k*localResMap_size[1])+k2)
			if kInd >= localResMap_out.shape[1]: continue

			if dimension == 3:
				for l2 in range(localResMap_size[2]):
					lInd = int((l*localResMap_size[2])+l2)
					if lInd >= localResMap_out.shape[2]: continue
					
					pVals = []
					# fscVals = []
					for m in range(len(locResMap)):
						pVals.append(locResMap[m][j2][k2][l2])

					qVals_FDR = pAdjust_BY(np.array(pVals))
					res_index = calcResIndex(np.copy(qVals_FDR), p_cutoff, test2)
					if res_index < 0: #only if first shell failed. 
						continue
					localResMap_out[jInd,kInd,lInd] = int((1/res[res_index])*100)/100 #np.round(1/(res[res_index]), 4)
			if dimension == 2:
				pVals = []
				for m in locResMap:
					pVals.append(m[j2][k2])
				
				qVals_FDR = pAdjust_BY(np.array(pVals))
				res_index = calcResIndex(np.copy(qVals_FDR), p_cutoff, test2)
				
				if res_index < 0: #only if first shell failed. 
					continue
				localResMap_out[jInd,kInd] = int((1/res[res_index])*100)/100 #np.round(1/(res[res_index]), 4)
	
	return localResMap_out


def localResolutionsGPU(collapseWindow_i, sample1_filtered, sample2_filtered, permutated_sample2_filtered, permutated_sample1_filtered, windowSize, window_size_i, indexes, corrected_box_size, maxWindow_half, stepSize, gpu_ids, bordersCurr, it_randomMaps, referenceDistSize):
	"""
	Local correlation measurements for GPU. Wrapper function.
	"""

	floatWindow = np.copy(windowSize) # The window size for determination of the bool array is the original float
	windowSize = int(np.ceil(windowSize)) # The window size for extracting the box has to be an integer value
	window = windowSize*2+1
	center = int(window/2)
	indices = np.indices([window for _ in range(len(sample1_filtered.shape))])
	distances = np.linalg.norm(indices - center, axis=0)
	bool_array = np.ones([window for _ in range(len(sample1_filtered.shape))])
	bool_array[distances>floatWindow] = 0

	if collapseWindow_i:
		middle_index = bool_array.shape[0] // 2  # Compute the middle index
		bool_array = bool_array[middle_index - window_size_i:middle_index + window_size_i + 1, :, :]
	else:
		window_size_i = windowSize

	start_initPermut = datetime.datetime.now()
	permuted_map = initial_permutations(bordersCurr, maxWindow_half, windowSize, window_size_i, permutated_sample1_filtered, permutated_sample2_filtered, bool_array, it_randomMaps, referenceDistSize)
	start_gpu = datetime.datetime.now()


	outputMap = np.zeros([len(range(maxWindow_half[i], maxWindow_half[i] + corrected_box_size[i], stepSize[i])) for i in range(len(corrected_box_size))], dtype=np.float32)
	results = []
	shape0 = int(outputMap.shape[0]/len(gpu_ids))
	rest = outputMap.shape[0]%len(gpu_ids)
	start_GPU = datetime.datetime.now()
	from concurrent.futures import ThreadPoolExecutor
	with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
		futures = []
		for gpu_id in range(len(gpu_ids)):
			dimsMin = int(gpu_id*shape0)
			minClipMap = ((gpu_id*shape0)*stepSize[0]) + maxWindow_half[0]
			maxClipMap = (((gpu_id+1)*shape0)*stepSize[0]) + maxWindow_half[0]

			if gpu_id == gpu_ids[-1]:
				dimsMax = int(((gpu_id+1)*shape0)+rest)
				maxClipMap = ((((gpu_id+1)*shape0)+rest)*stepSize[0]) + maxWindow_half[0]
			else:
				dimsMax = int(((gpu_id+1)*shape0))
			results.append(np.array(np.copy(outputMap[dimsMin:dimsMax])))

			sample1_filtered_clipped = sample1_filtered[minClipMap-maxWindow_half[0]:maxClipMap+maxWindow_half[0]]
			sample2_filtered_clipped = sample2_filtered[minClipMap-maxWindow_half[0]:maxClipMap+maxWindow_half[0]]
			corrected_box_size_corrected = np.copy(corrected_box_size)
			corrected_box_size_corrected[0] = maxClipMap - minClipMap
			futures.append(executor.submit(runLocal_cuda, corrected_box_size_corrected, maxWindow_half, windowSize, window_size_i, sample1_filtered_clipped, sample2_filtered_clipped,
					stepSize, permuted_map, bool_array, start_gpu, gpu_ids[gpu_id], results[gpu_id], dimsMin, dimsMax))
			
		# Wait for all computations to finish
		for future in futures:
			future.result()
	fsc_map = results[0]
	# print("onlyGPU " + str(datetime.datetime.now()-start_GPU) + "\n")

	for x in range(1, len(gpu_ids)):
		fsc_map = np.concatenate((fsc_map, results[x]), axis=0)

	pValueMap = np.array(fsc_map, dtype=np.float16)
	
	# print("gpu function overall took " + str(datetime.datetime.now() - start_gpu) + "\n")
	return pValueMap
 
 
def localResolutions(queue, collapseWindow_i, sample1_filtered, sample2_filtered, permutated_sample2_filtered, permutated_sample1_filtered, windowSize, window_size_i, indexes, bordersCurr, it_randomMaps, referenceDistSize, corrected_box_size, maxWindow_half, stepSize):
	"""
	Local correlation measurements for CPU, multiple threads. Wrapper function.
	"""		

	floatWindow = np.copy(windowSize) #the window size for determination of the bool array is the original float
	windowSize = int(np.ceil(windowSize)) #the window size for extracting the box has to be an integer value
	window = windowSize*2+1
	center = int(window/2)
	indices = np.indices([window for _ in range(len(sample1_filtered.shape))])
	distances = np.linalg.norm(indices - center, axis=0)
	bool_array = np.ones([window for _ in range(len(sample1_filtered.shape))])
	bool_array[distances>floatWindow] = 0
	# if window_size_i == 1:
	if collapseWindow_i:
		middle_index = bool_array.shape[0] // 2  # Compute the middle index
		bool_array = bool_array[middle_index - window_size_i:middle_index + window_size_i + 1, :, :]  # Shape (1, x, y) 
	else:
		window_size_i = windowSize

	permuted_map = initial_permutations(bordersCurr, maxWindow_half, windowSize, window_size_i, permutated_sample1_filtered, permutated_sample2_filtered, bool_array, it_randomMaps, referenceDistSize)

	# permuted_map = initial_permutations(bordersCurr, maxWindow_half, windowSize, window_size_i, sample1_filtered, permutated_sample2_filtered, bool_array, it_randomMaps, referenceDistSize)

	pValsMap = loopOverMap(corrected_box_size, maxWindow_half, windowSize, window_size_i, sample1_filtered, sample2_filtered, stepSize, permuted_map, bool_array)
	
	# queue.put([indexes, fsc_map])       

	queue.put([indexes, pValsMap])       

 
 
	

def initial_permutations(bordersCurr, maxWindow_half, windowSize, window_size_i, paddedHalfMap1_array, paddedHalfMap2_array, 
							bool_array, it_randomMaps, referenceDistSize):
	"""
	Calculation permutations.
	"""		
	iterator_maps = 0
	count = 0
	permutedCorCoeffs = []
	while True:
		paddedHalfMap1 = paddedHalfMap1_array[0] # first half map
		paddedHalfMap2 = paddedHalfMap2_array[iterator_maps] # permutated half-maps
		shapeCut = paddedHalfMap2_array[0].shape

		indices1 = [np.random.randint((maxWindow_half[i]) + bordersCurr[i][0], bordersCurr[i][1] - (maxWindow_half[i]) - 1) for i in range(len(shapeCut))]
		slices1 = [slice(index_i - windowSize, index_i + windowSize + 1) for index_i in indices1]
  
		indices2 = [np.random.randint((maxWindow_half[i]) + bordersCurr[i][0], bordersCurr[i][1] - (maxWindow_half[i]) - 1) for i in range(len(shapeCut))]
		slices2 = [slice(index_i - windowSize, index_i + windowSize + 1) for index_i in indices2]


		slices1[0] = slice(indices1[0] - window_size_i, indices1[0] + window_size_i + 1)
		slices2[0] = slice(indices2[0] - window_size_i, indices2[0] + window_size_i + 1)

  
		windowHalfmap1 = paddedHalfMap1[tuple(slices1)]
		windowHalfmap1 = windowHalfmap1[bool_array == 1]

		windowHalfmap2 = paddedHalfMap2[tuple(slices2)]
		windowHalfmap2 = windowHalfmap2[bool_array == 1]
		
		FSC = correlationCoefficient(windowHalfmap1, windowHalfmap2)
		permutedCorCoeffs = np.append(permutedCorCoeffs, FSC)
		if count == (referenceDistSize-1): break #iterations

		count += 1
		if count%np.ceil(referenceDistSize/it_randomMaps) == 0:
			iterator_maps += 1
   
	return permutedCorCoeffs
		
		
def loopOverMap(corrBoxSize, maxWindow_half, window_size, window_size_i, paddedHalfMap1, paddedHalfMap2,
							stepSize, permuted_map, bool_array):
	"""
	Local correlation measurements for CPU, multiple threads. Executing function.
	"""		

	p_map = np.zeros([len(range(maxWindow_half[i], maxWindow_half[i] + corrBoxSize[i], stepSize[0])) for i in range(len(corrBoxSize))], dtype=np.float32)
	# fsc_map = np.copy(p_map)

	permutedCorCoeffs = []
	dim = len(paddedHalfMap1.shape)
	iInd = 0
	for i in range(maxWindow_half[0], maxWindow_half[0] + corrBoxSize[0], stepSize[0]): 
		jInd = 0
		for j in range(maxWindow_half[1], maxWindow_half[1] + corrBoxSize[1], stepSize[1]):
			if dim == 3:

				kInd = 0
				for k in range(maxWindow_half[2], maxWindow_half[2] + corrBoxSize[2], stepSize[2]):
					indices = [i,j,k]
					slices = [slice(ind_ind - window_size, ind_ind + window_size + 1) for ind_ind in indices]
					slices[0] = slice(i - window_size_i, i + window_size_i + 1) #if we want to collapse! otherwise, window_size_i = window_size
					window_halfmap1 = paddedHalfMap1[tuple(slices)]
					window_halfmap1 = window_halfmap1[bool_array == 1]
					window_halfmap2 = paddedHalfMap2[tuple(slices)]
					window_halfmap2 = window_halfmap2[bool_array == 1]
					fsc_vals = correlationCoefficient(window_halfmap1, window_halfmap2)
					pVal = (permuted_map[permuted_map > fsc_vals].shape[0])/(float(permuted_map.shape[0]))
					p_map[iInd, jInd, kInd] = pVal
					# fsc_map[iInd, jInd, kInd] = fsc_vals[0]
					kInd += 1

			if dim == 2:
				indices = [i,j]
				slices = [slice(ind_ind - window_size, ind_ind + window_size +1) for ind_ind in indices]
				window_halfmap1 = paddedHalfMap1[tuple(slices)]
				window_halfmap1 = window_halfmap1[bool_array == 1]
				window_halfmap2 = paddedHalfMap2[tuple(slices)]
				window_halfmap2 = window_halfmap2[bool_array == 1]
				fsc_vals = correlationCoefficient(window_halfmap1, window_halfmap2)
				pVal = (permuted_map[permuted_map > fsc_vals].shape[0])/(float(permuted_map.shape[0]))
				p_map[iInd, jInd] = pVal
				# fsc_map[iInd, jInd] = fsc_vals[0]
	
	
			jInd += 1                
		iInd += 1
	return np.array(p_map).astype(np.float16)


def get_gpu_memory():
	from numba import cuda
	"""Get current GPU memory usage in MB"""
	meminfo = cuda.current_context().get_memory_info()
	used = (meminfo.total - meminfo.free) / (1024**2)  # Convert to MB
	return used