import mrcfile
import numpy as np
import os
import functools
import multiprocessing as mp
import datetime
import pyfftw
import gc
import scripts.utils_resolve as utils_resolve
import datetime

def main(mode, config, apix, odd_input, even_input, cpu_threads, gpu_enabled, gpu_settings, run_fast, signal_mask_input, mask_measure, outputDir, inputDir):
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)
  
	# Some configurations for the different input types
	if config == "Refined-Maps":
		preAddToName = ""
		runOnAveragedMap = True # SPA/STA outputs (True) or tomograms/micrographs/tilt-series (False)
		collapseWindow_i = False
		window_size_i = 0	
		test2 = False	
		p_cutoff = 0.01 # q cutoff value
	if config == "Micrographs":
		preAddToName = ""
		runOnAveragedMap = False  # SPA/STA outputs (True) or tomograms/micrographs/tilt-series (False)
		collapseWindow_i = False 
		window_size_i = 0
		test2 = True # Allowing one shell to cross q-threshold if the next shell is below
		p_cutoff = 0.05 # q cutoff value
	if config == "Tilt-Series":
		preAddToName = ""
		runOnAveragedMap = False  # SPA/STA outputs (True) or tomograms/micrographs/tilt-series (False)
		collapseWindow_i = True # Collapse window over z-dimension for tilt-series (so that radius will by (x,y,1))
		window_size_i = 0
		test2 = True # Allowing one shell to cross q-threshold if the next shell is below
		p_cutoff = 0.05 # q cutoff value
	if config == "Tomograms":
		preAddToName = ""
		runOnAveragedMap = False  # SPA/STA outputs (True) or tomograms/micrographs/tilt-series (False)
		collapseWindow_i = False
		window_size_i = 0
		test2 = True # Allowing one shell to cross q-threshold if the next shell is below
		p_cutoff = 0.05 # q cutoff value

	# Inputs
	resMax = 10 # Check shells up to 10*Nyquist 
	accuracy_steps = 1 # Defines the Fourier-space sampling
	referenceDistSize = 10000 # Size of reference distribution
	input_gpuIds = gpu_settings # If left empty, choose by first two. Otherwise, enter ids.
	numCores = cpu_threads
	printDebugging = False
	boxValue = "max" # Set a fixed value or "max". (fixed value in case memory unexpectedly small)
	filterChoice = utils_resolve.hypTan # Define filter
	spacingFilter = 0.05 # Size of shells
	falloff = 1.5 # falloff for hypTan bandpass filter
	if run_fast:
			accuracy_steps = 2
	# Configurations for running on GPU
	gpu_ids = []
	if gpu_enabled:
		try:
			from numba import cuda
			runOnGPU = 1
			if len(input_gpuIds) == 0:
				gpu_ids = [gpu.id for gpu in cuda.gpus][:2]
			else:
				gpu_ids = list(np.array(input_gpuIds.split(",")).astype(int))
				for check_gpu in gpu_ids:
					if check_gpu not in [gpu.id for gpu in cuda.gpus]:
						print("Cannot find GPU. Exit.")
						return
			if len(gpu_ids) > 2:
				print("Warning: Using more than 2 GPUs is not recommended, as it may slow down processing.")
			if config == "Micrographs":
				print("Warning: GPU usage is not recommended for micrographs, as it will likely slow down processing.")

			print("running in GPU mode with GPU(s): " + " ".join(np.array(gpu_ids).astype(str)))
		except: 
			runOnGPU = 0
			gpu_ids = [0]
			print("could not import numba, running in CPU mode instead")
	else:
		runOnGPU = 0
		gpu_ids = [0]
		print("running in CPU mode")	
	if (runOnGPU >= 1): # Correct filter function (input gpu function if GPU is used)
		filterChoice = getattr(utils_resolve, filterChoice.__name__+"_cuda") 
	
	# Configuring batch mode for running on datasets - running on any number of files in the given directory
	it_loops = 1
	if mode == "batch":
		odd_id = odd_input
		even_id = even_input
		matching_files = [f for f in os.listdir(inputDir) if odd_id in f]
		it_loops = len(matching_files)
  
	
	# Starting to loop over all files to process
	nameArray, resGlobArray, ratioSignalArray =[], [], []
	for iterate_files in range(it_loops):
     
		start_total = datetime.datetime.now()
		if mode == "batch":
			odd_input = matching_files[iterate_files]
			even_input = odd_input.replace(odd_id, even_id)
			odd_input = os.path.join(inputDir, odd_input)
			even_input = os.path.join(inputDir, even_input)

		# Safety checks
		if not odd_input.endswith((".mrc", ".map")):
			continue
		if odd_input == even_input:
			print("Error: given same path twice. End.")
			return

		# Naming file
		preAddToName = odd_input.split("/")[-1][:-4] + "_" + config + "_locRes"  
		# print("CHECK IF FILE ALREADY PROCESSED. IF SO, SKIP")
		from pathlib import Path
		outputFilename_LocRes = os.path.join(outputDir, preAddToName + ".mrc")
		if Path(outputFilename_LocRes).exists():
			print("Warning: " + outputFilename_LocRes + " already exists. This file is processed already. For reprocessing, please delete output file or define new output directory. SKIP!\n")
			continue

		# Initializations and reading data
		halfMap1 = mrcfile.open(odd_input, mode='r')
		halfMap2 = mrcfile.open(even_input, mode='r')
		# afterReading = datetime.datetime.now()
		# print("reading taking " + str(afterReading-start_total) + "\n")
		print("\nusing input half-maps: ")
		print(even_input)
		print(odd_input)
		print("")
		halfMap1Data = halfMap1.data
		halfMap2Data = halfMap2.data
		sizeMap = halfMap1Data.shape
		dimension = len(sizeMap)

		# Some more safety checks
		if config == "Refined-Maps":
			if dimension != 3:
				print("Error: inputs should be 3D")
				return
		if config == "Micrographs":
			if dimension != 2:
				print("Error: inputs should be 2D")
				return    
		if config == "Tilt-Series":
			if dimension != 3:
				print("Error: inputs should be 3D")
				return    
		if config == "Tomograms":
			if dimension != 3:
				print("Error: inputs should be 3D")
				return
		if halfMap1Data.shape != halfMap2Data.shape:
			print("input maps do not have same size. Exit.")
			return

		# Configuring step size dependent on input data dimensions
		if dimension == 2:
			stepSize = [5,5]
		else:
			stepSize = [2,2,2]
			if run_fast:
				stepSize = [3,3,3]

		# Processing signal mask for median estimate
		if (len(signal_mask_input) == 0) or (not os.path.exists(signal_mask_input)):
			signal_mask = None
		else:
			signal_mask = mrcfile.open(signal_mask_input).data*1
			signal_mask[signal_mask < 1] = 0
			signal_mask[signal_mask >= 1] = 1
			signal_mask = np.array(signal_mask, dtype=bool)
			print("using signal mask for median estimate: " + str(signal_mask_input))

		# Reading pixel size
		print("Input configurations_____________")
		if apix is None:
			apix = np.round(float((halfMap1.voxel_size).x),2)
			apix_y = np.round(float((halfMap1.voxel_size).y),2)
			if dimension == 2:
				print("pixel size read from header (x,y): " + str(apix) + " " + str(apix_y))
			if dimension == 3:
				apix_z = np.round(float((halfMap1.voxel_size).z),2)
				print("pixel size read from header (x,y,z): " + str(apix) + " " + str(apix_y) + " " + str(apix_z)) # Z-value may differ for Tilt-series
		lowRes = resMax*apix # Lowest resolution 
		lowResMax = 1/(np.fft.rfftfreq(np.min(sizeMap))[1]/apix)
		lowRes = np.min([lowRes, lowResMax])
		print("lowest resolution to consider (10*apix): " + str(np.round(lowRes,2)))
		del halfMap1 # Cleaning
		del halfMap2 # Cleaning
  
		# This is for tilt-series (collapse window refers to collapsing z-radius to 1)
		if collapseWindow_i:
			dimension_windows = 2
			stepSize = [5,5,5]
			stepSize[0] = 1 # Collapse z to 1. Not that for numpy arrays, x and z are swapped (z,y,x instead of x,y,z)
		else:
			dimension_windows = dimension
		print("using step size: " + str(" ".join(np.array(stepSize[::-1]).astype(str)))) # Adjust x-z swap

		# Get windows (radii) and shells
		sizeVol = 100
		shells_dict = utils_resolve.calculateShells(sizeVol, apix, lowRes, spacingFilter, accuracy_steps)
		shells = [(np.array(v)) for k,v in shells_dict.items()]
		resolutions = [(k) for k,v in shells_dict.items()]
		shellStr = ""
		for i in shells:
			shellStr += str(np.round(1/i[0],3)) + "-" + str(np.round(1/i[1],3)) + "; "

		# Get windows and box sizes from precalculated empirical simulations. Print out parameters used.
		windows = utils_resolve.getWindowsEmpirical(np.array(resolutions)*apix, dimension_windows)
		maxWindow_half = [int(np.ceil(np.max(windows)))+1, int(np.ceil(np.max(windows)))+1, int(np.ceil(np.max(windows)))+1]
		maxWindow_half = maxWindow_half[:dimension]
		boxSize, corrected_box_size = utils_resolve.calculateEfficientBoxSize(sizeMap, boxValue, maxWindow_half, runOnGPU, dimension, collapseWindow_i) # box size
		blueprint_box = np.zeros(boxSize, dtype=np.float32) 
		localResMap_blueprint = np.zeros([len(range(0, corrected_box_size[i], stepSize[i])) for i in range(len(corrected_box_size))], dtype=np.float16)
		localResMap_out = np.zeros([len(range(0, sizeMap[i], stepSize[i])) for i in range(len(sizeMap))], dtype=np.float32)  
		localResMap_out.fill(lowRes)
		localResMap_size = localResMap_blueprint.shape
		# print("init 1 taking " + str(afterInit1-start_total) + "\n")
		print("using window radii [pix]: " + " ".join(map(str,np.round(windows,1))))
		print("to measure resolutions [Å]: " + " ".join(map(str,np.round(1/np.array(resolutions),2))))
		# print("using shells: " + shellStr)
		print("")
 
		# Get frequency maps (for later bandpass filtering)
		if collapseWindow_i:
			freqMap = utils_resolve.calculate_frequency_map(boxSize[1:])/float(apix)
		else:
			freqMap = utils_resolve.calculate_frequency_map(boxSize)/float(apix)


		# Old functionality, previously serving as a backup in case maps are too large to fit in memory. Unrelevant, only kept for potential later use cases.
		iterate_boxSize = (np.ceil(sizeMap/corrected_box_size)).astype(int) # determine iterations for given boxSize minus half max window size
		overallBoxes = np.prod(iterate_boxSize) # Set to 1 always
		# print("calculate resolutions for " + str(overallBoxes) + " box(es)")
		# print("using adjusted box Size " + " ".join(map(str, boxSize)))
		# afterInit1 = datetime.datetime.now()

		# Define pyfftw 
		pyfftwSize = [i for i in boxSize]
		if collapseWindow_i:
			pyfftwSize = pyfftwSize[1:]
			maxWindow_half[0] = 0
		res_obj_inv = 0
		pyfftw_numCores = 1
		if runOnGPU: # When multi-threading on CPU, it seems one thread for FFT is best (as correlation of local boxes from previous shell is computed in parallel)
			pyfftw_numCores = numCores
		pyfftw.config.NUM_THREADS = pyfftw_numCores # Enable multithreading and caching
		pyfftw.interfaces.cache.enable()
		pyfftwMap = pyfftw.empty_aligned(pyfftwSize, dtype='float32')
		pyfftwMap[:] = np.random.normal(1.0, 0.1, size=pyfftwSize)
		output_shape = [i for i in pyfftwSize]
		output_shape[-1] = output_shape[-1]//2+1
		fft_output = pyfftw.empty_aligned(output_shape, dtype='complex64')
		res_obj = pyfftw.builders.rfftn(
			pyfftwMap,
			threads=pyfftw_numCores,
			planner_effort='FFTW_MEASURE',
			avoid_copy=True,
			auto_align_input=True,
			auto_contiguous=True
		)
		res_obj_inv = pyfftw.builders.irfftn(
			fft_output,
			s=pyfftwSize, 
			threads=pyfftw_numCores,
			planner_effort='FFTW_MEASURE',
			avoid_copy=True,
			auto_align_input=True,
			auto_contiguous=True
		)	
		afterFFTInit = datetime.datetime.now()
		# print("init 2 FFT taking " + str(afterFFTInit-afterInit1) + "\n")
 		
  
		# Padding up to half the correlation calculating window radius with noise for tomograms, micrographs and tilt-series - otherwise, edges will have high resolution
		# Note that padding up to lowest effecient box size for FFT will still be done with zeros
		if runOnAveragedMap: # For refined maps
			noise_padding = False
		else: 
			noise_padding = True
		if not noise_padding:
			padded_inputMap_1 = np.zeros([sizeMap[i] + 2 * maxWindow_half[i] for i in range(len(sizeMap))])
			padded_inputMap_2 = np.copy(padded_inputMap_1)
		else:
			shapePadded = [sizeMap[i] + 2 * maxWindow_half[i] for i in range(len(sizeMap))]
			padded_inputMap_1 = np.random.choice(halfMap1Data.flatten(), size=np.prod(shapePadded)).reshape(shapePadded)
			padded_inputMap_2 = np.random.choice(halfMap2Data.flatten(), size=np.prod(shapePadded)).reshape(shapePadded)


		# Place half-maps into padded maps
		slices = [slice(maxWindow_half[i],maxWindow_half[i]+sizeMap[i]) for i in range(len(sizeMap))]
		padded_inputMap_1[tuple(slices)] = halfMap1Data
		padded_inputMap_2[tuple(slices)] = halfMap2Data
		sizeMap_padded = padded_inputMap_1.shape 
		if signal_mask is not None:
			signalMaskPadded = np.zeros([sizeMap[i] + 2 * maxWindow_half[i] for i in range(len(sizeMap))], dtype=bool)
			signalMaskPadded[tuple(slices)] = signal_mask
		else:
			signalMaskPadded = None


		# CPU multi-threading processing for running on cpu and in case filling on GPU fails, CPU multi-threading function for filling.
		partial_locaRes = None
		partial_fillMap = None
		if runOnGPU < 1:
			partial_locaRes = functools.partial(utils_resolve.localResolutions, corrected_box_size=corrected_box_size, maxWindow_half=maxWindow_half, stepSize=stepSize)
		partial_fillMap = functools.partial(utils_resolve.fillMapMultiThread, p_cutoff=p_cutoff, test2=test2, dimension=dimension)


		# Outdated, ignore. Prepare boxes if input is too large for memory
		innerIt = 1
		if dimension == 3: innerIt = iterate_boxSize[2]
		boxes_iterate = []
		for j in range(iterate_boxSize[0]):
			for k in range(iterate_boxSize[1]):
				for l in range(innerIt):
					boxes_iterate.append([j,k,l])
	
				
		# How many random maps are needed to get a good enough reference distribution
		windowtest = np.max(maxWindow_half)*2+1 
		maxEntries = np.prod([windowtest for _ in range(dimension)])	
		if collapseWindow_i:
			maxEntries = np.prod([windowtest for _ in range(dimension_windows)])	
		possibleTests_nonOverlapping = int(np.prod(corrected_box_size)/maxEntries)
		possibleTests_nonOverlapping = possibleTests_nonOverlapping**2 # Consider enhanced possibility space, dependencies are introduced in Fourier space, and are thus real-space location independent 
		it_randomMaps = int(np.ceil(referenceDistSize / possibleTests_nonOverlapping))
		# print("use n random maps: " + str(it_randomMaps))

	
		# some debugging saves
		if printDebugging:
			saveMask1 = mrcfile.new(os.path.join("debugging", "autoMasks", "masked" + odd_input.split("/")[-1]), overwrite=True)
			mapToSave = np.float32(signal_mask)
			saveMask1.set_data(mapToSave)
			saveMask1.voxel_size = apix
			saveMask1.close()
			del mapToSave
	
	
		# Correlation calculations
		afterinit3 = datetime.datetime.now()
		# print("init 3 taking " + str(afterinit3-afterFFTInit) + "\n")
		localResMap_out, actualRes_global_new, ratioSignal =  utils_resolve.iterateBoxesWindows(collapseWindow_i, localResMap_out, boxes_iterate, dimension, windows, window_size_i, blueprint_box, sizeMap_padded, boxSize, resolutions, filterChoice, apix, slices, padded_inputMap_1, padded_inputMap_2, res_obj, res_obj_inv, freqMap, shells, falloff, gpu_ids, runOnGPU, it_randomMaps, partial_locaRes, printDebugging, corrected_box_size, maxWindow_half, stepSize, referenceDistSize, numCores, localResMap_size, p_cutoff, test2, lowRes, partial_fillMap, signalMaskPadded, mask_measure, config, outputDir, runOnAveragedMap, preAddToName)
		resGlobArray.append(actualRes_global_new)
		ratioSignalArray.append(ratioSignal)
		nameArray.append(preAddToName)
	
	
		# Interpolating
		# As we never use every pixel/voxel, the grid needs to be interpolated in the end.
		localResMap_out = np.array(localResMap_out, dtype=np.float32)
		del blueprint_box
		localResMap_out[localResMap_out>lowRes] = lowRes   
		gc.collect() # Make sure garbage is collected before interpolating to free memory
		print("interpolating grid")
		start_interpolate = datetime.datetime.now()
		if np.max(stepSize) != 1: # This is always the case in this default script
			if collapseWindow_i: # For tilt-series
				localResMap = []
				for i in range(localResMap_out.shape[0]):
					localResMap.append(utils_resolve.interpolate_with_zoom(localResMap_out[i], sizeMap[1:], stepSize, lowRes))
				localResMap = np.array(localResMap)
			else:
				if (np.prod(sizeMap)<(700**3)): 
					localResMap = utils_resolve.interpolate_with_zoom(localResMap_out, sizeMap, stepSize, lowRes)
				else: # for very large maps, interpolating chunk-wise.
					localResMap = utils_resolve.interpolate_chunks(localResMap_out, sizeMap, dimension, iterate_boxSize, localResMap_size, localResMap_out.shape, stepSize, [500,500,500])
		else:
			localResMap = np.copy(localResMap_out)    
		# print("interpolation took " + str(datetime.datetime.now()-start_interpolate) + "\n")
		del localResMap_out 
		localResMap[np.isnan(localResMap)] = lowRes
		# localResMap = np.where(np.isnan(localResMap), lowRes, localResMap) 


		# Write output
		print("write here: " + str(outputFilename_LocRes))
		localResMapMRC = mrcfile.new(outputFilename_LocRes, overwrite=True)
		localResMap = np.float32(localResMap)
		localResMapMRC.set_data(localResMap)
		if config == "Tilt-Series":
			localResMapMRC.voxel_size = (apix,apix,apix_z)
		else:
			localResMapMRC.voxel_size = apix
		localResMapMRC.close()
		end_total = datetime.datetime.now()
		print("IN TOTAL: " + str(end_total-start_total) + "\n\n")
		
  
		# For 2D calculations (micrographs), also give a 2D image as output.
		if dimension==2:
			import matplotlib.pyplot as plt
			localResMap = np.flipud(localResMap) #stuff read in numpy from .mrc is axis swapped. This does not matter when saving as .mrc again, but it does here.
			plt.rcParams['font.size'] = 16
			cmap = plt.get_cmap('bwr') # blue -> red
			fig, ax = plt.subplots() 
			img = ax.imshow(localResMap, cmap=cmap, vmin=2*apix, vmax=lowRes)
			cbar = plt.colorbar(img, ax=ax, pad=0.05, aspect=20)
			cbar.set_label('Resolution') 
			cbar.ax.invert_yaxis()
			num_ticks = 6
			cbar_ticks = np.linspace(2*apix, lowRes, num_ticks)  # Evenly spaced ticks between vmin and vmax
			cbar.set_ticks(cbar_ticks)
			cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
			plt.tight_layout()
			plt.axis('off')
			plt.savefig(outputFilename_LocRes[:-3]+"png", bbox_inches='tight', pad_inches=0.05) 


	# For batch mode, write summary.tsv file for all the processed files
	if mode == "batch":
		with open(os.path.join(outputDir, "summary.tsv"), 'w', encoding='utf-8') as file:
			# Write header
			file.write('id\tmedian_resolution\tsignal_ratio\n')
			nameArray = np.array(nameArray)
			resGlobArray = np.array(resGlobArray)
			ratioSignalArray = np.array(ratioSignalArray)
			sortedIndices = np.argsort(resGlobArray)
			nameArray = nameArray[sortedIndices]
			resGlobArray = resGlobArray[sortedIndices]
			ratioSignalArray = ratioSignalArray[sortedIndices]
			# Write data rows
			for i in range(len(nameArray)):
				file.write(f'{nameArray[i]}\t{resGlobArray[i]}\t{ratioSignalArray[i]}\n')



if __name__ == '__main__':    
	main()
