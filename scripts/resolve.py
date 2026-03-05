import mrcfile
import numpy as np
import os
import functools
import multiprocessing as mp
import datetime
import pyfftw
import gc
import scripts.utils_resolve as utils_resolve


def process_one_file(args):
	"""Process a single file pair in a worker process. Returns (name, resolution, signal_ratio) or None."""
	(mode, config, apix, odd_input, even_input, cpu_threads, gpu_enabled, gpu_settings,
	 run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir, inputDir,
	 runOnAveragedMap, collapseWindow_i, window_size_i, test2, p_cutoff,
	 resMax, accuracy_steps, referenceDistSize, printDebugging, boxValue,
	 spacingFilter, falloff) = args

	filterChoice = utils_resolve.hypTan

	# Configurations for running on GPU
	gpu_ids = []
	numCores = cpu_threads
	input_gpuIds = gpu_settings
	if gpu_enabled:
		try:
			from numba import cuda
			runOnGPU = 1
			if len(input_gpuIds) == 0:
				gpu_ids = [gpu.id for gpu in cuda.gpus][:2]
			else:
				gpu_ids = input_gpuIds 
				for check_gpu in gpu_ids:
					if check_gpu not in [gpu.id for gpu in cuda.gpus]:
						print("Cannot find GPU. Exit.")
						return None
			print("running in GPU mode with GPU(s): " + " ".join(np.array(gpu_ids).astype(str)))
		except: 
			print("WARNING: could not import numba. Cannot run in GPU mode!")
			return
	else:
		runOnGPU = 0
		gpu_ids = [0]
		if mode != "batch" or runOnGPU: print("running in CPU mode")	
	if (runOnGPU >= 1):
		filterChoice = getattr(utils_resolve, filterChoice.__name__+"_cuda") 

	# Safety checks
	if not odd_input.endswith((".mrc", ".map")):
		return None
	if odd_input == even_input:
		print("Error: given same path twice. End.")
		return None

	# Naming file
	preAddToName = odd_input.split("/")[-1][:-4] + "_" + config + "_locRes"  
	outputFilename_LocRes = os.path.join(outputDir, preAddToName + ".mrc")

	# Initializations and reading data
	halfMap1 = mrcfile.open(odd_input, mode='r')
	halfMap2 = mrcfile.open(even_input, mode='r')
	if mode != "batch" or runOnGPU:
		print("\nusing input half-maps: ")
		print(even_input)
		print(odd_input)
		print("")
	halfMap1Data = halfMap1.data
	halfMap2Data = halfMap2.data
	sizeMap = halfMap1Data.shape
	dimension = len(sizeMap)

	# More safety checks
	if config == "Refined-Maps":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None
	if config == "Micrographs":
		if dimension != 2:
			print("Error: inputs should be 2D")
			return None    
	if config == "Tilt-Series":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None    
	if config == "Tomograms":
		if dimension != 3:
			print("Error: inputs should be 3D")
			return None
	if halfMap1Data.shape != halfMap2Data.shape:
		print("input maps do not have same size. Exit.")
		return None

	# Configuring step size dependent on input data dimensions
	if dimension == 2:
		stepSize = [5,5]
	else:
		stepSize = [2,2,2]
		if run_fast:
			stepSize = [3,3,3]

	# Processing signal mask for median estimate
	signal_mask = None
	if mask_strategy == "signal_mask":
		if (len(signal_mask_input) == 0) or (not os.path.exists(signal_mask_input)):
			signal_mask = None
		else:
			signal_mask = mrcfile.open(signal_mask_input).data*1
			signal_mask[signal_mask < 1] = 0
			signal_mask[signal_mask >= 1] = 1
			signal_mask = np.array(signal_mask, dtype=bool)
			print("using signal mask for median estimate: " + str(signal_mask_input))
		if signal_mask is None:
			print("Signal mask not found. Default to remove background strategy for median resolution.")
		if signal_mask.shape != halfMap1Data.shape:
			print("Signal mask has not shape of input map. Default to remove background strategy for median resolution.")
			signal_mask = None
	if mask_strategy == "full_map":
		signal_mask = np.ones(halfMap1Data.shape)

	# Reading pixel size
	if mode != "batch" or runOnGPU: print("Input configurations_____________")
	if apix is None:
		apix = np.round(float((halfMap1.voxel_size).x),2)
		apix_y = np.round(float((halfMap1.voxel_size).y),2)
		if dimension == 2:
			if mode != "batch" or runOnGPU: print("pixel size read from header (x,y): " + str(apix) + " " + str(apix_y))
		if dimension == 3:
			apix_z = np.round(float((halfMap1.voxel_size).z),2)
			if mode != "batch" or runOnGPU: print("pixel size read from header (x,y,z): " + str(apix) + " " + str(apix_y) + " " + str(apix_z)) # Z-value may differ for Tilt-series
	lowRes = resMax*apix # Lowest resolution 
	lowResMax = 1/(np.fft.rfftfreq(np.min(sizeMap))[1]/apix)
	lowRes = np.min([lowRes, lowResMax])
	if mode != "batch" or runOnGPU: print("lowest resolution to consider (10*apix): " + str(np.round(lowRes,2)))

	# This is for tilt-series (collapse window refers to collapsing z-radius to 1)
	if collapseWindow_i:
		dimension_windows = 2
		stepSize = [5,5,5]
		stepSize[0] = 1 # Collapse z to 1. Not that for numpy arrays, x and z are swapped (z,y,x instead of x,y,z)
	else:
		dimension_windows = dimension
	if mode != "batch" or runOnGPU: print("using step size: " + str(" ".join(np.array(stepSize[::-1]).astype(str)))) # Adjust x-z swap

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
	if mode != "batch" or runOnGPU: 
		print("using window radii [pix]: " + " ".join(map(str,np.round(windows,1))))
		print("to measure resolutions [Å]: " + " ".join(map(str,np.round(1/np.array(resolutions),2))))
		print("")

	# Get frequency maps (for bandpass filtering)
	if collapseWindow_i:
		freqMap = utils_resolve.calculate_frequency_map(boxSize[1:])/float(apix)
	else:
		freqMap = utils_resolve.calculate_frequency_map(boxSize)/float(apix)


	# Outdated. Previously serving as a backup in case maps are too large to fit in memory.
	iterate_boxSize = (np.ceil(sizeMap/corrected_box_size)).astype(int) # determine iterations for given boxSize minus half max window size
	overallBoxes = np.prod(iterate_boxSize) # Currently always 1

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
	rng = np.random.default_rng(seed=42)
	pyfftwMap[:] = rng.normal(1.0, 0.1, size=pyfftwSize)
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
		padded_inputMap_1 = rng.choice(halfMap1Data.flatten(), size=np.prod(shapePadded)).reshape(shapePadded)
		padded_inputMap_2 = rng.choice(halfMap2Data.flatten(), size=np.prod(shapePadded)).reshape(shapePadded)


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

	halfMap1.close()
	halfMap2.close()
	del halfMap1Data, halfMap2Data
	# CPU multi-threading processing for running on cpu and in case filling on GPU fails, CPU multi-threading function for filling.
	partial_locaRes = None
	partial_fillMap = None
	if runOnGPU < 1:
		partial_locaRes = functools.partial(utils_resolve.localResolutions, corrected_box_size=corrected_box_size, maxWindow_half=maxWindow_half, stepSize=stepSize)
	partial_fillMap = functools.partial(utils_resolve.fillMapMultiThread, p_cutoff=p_cutoff, test2=test2, dimension=dimension)


	# Outdated. Previously used to prepare boxes if input is too large for memory
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


	# some debugging saves
	if printDebugging:
		saveMask1 = mrcfile.new(os.path.join("debugging", "autoMasks", "masked" + odd_input.split("/")[-1]), overwrite=True)
		mapToSave = np.float32(signal_mask)
		saveMask1.set_data(mapToSave)
		saveMask1.voxel_size = apix
		saveMask1.close()
		del mapToSave


	# Correlation calculations
	localResMap_out, actualRes_global_new, ratioSignal =  utils_resolve.iterateBoxesWindows(mode, collapseWindow_i, localResMap_out, boxes_iterate, dimension, windows, window_size_i, blueprint_box, sizeMap_padded, boxSize, resolutions, filterChoice, apix, slices, padded_inputMap_1, padded_inputMap_2, res_obj, res_obj_inv, freqMap, shells, falloff, gpu_ids, runOnGPU, it_randomMaps, partial_locaRes, printDebugging, corrected_box_size, maxWindow_half, stepSize, referenceDistSize, numCores, localResMap_size, p_cutoff, test2, lowRes, partial_fillMap, signalMaskPadded, mask_measure, config, outputDir, runOnAveragedMap, preAddToName)


	# Interpolating
	# The grid needs to be interpolated in the end.
	localResMap_out = np.array(localResMap_out, dtype=np.float32)
	del blueprint_box
	localResMap_out[localResMap_out>lowRes] = lowRes   
	gc.collect() # Make sure garbage is collected before interpolating to free memory
	if mode != "batch" or runOnGPU: print("interpolating grid")
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
	del localResMap_out 
	localResMap[np.isnan(localResMap)] = lowRes


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
	

	# For 2D calculations (micrographs), also save 2D image as output.
	if dimension==2:
		import matplotlib.pyplot as plt
		localResMap = np.flipud(localResMap) # Correct axis swap for coherent visualization.
		plt.rcParams['font.size'] = 16
		cmap = plt.get_cmap('bwr')
		fig, ax = plt.subplots() 
		img = ax.imshow(localResMap, cmap=cmap, vmin=2*apix, vmax=lowRes)
		cbar = plt.colorbar(img, ax=ax, pad=0.05, aspect=20)
		cbar.set_label('Resolution') 
		cbar.ax.invert_yaxis()
		num_ticks = 6
		cbar_ticks = np.linspace(2*apix, lowRes, num_ticks)
		cbar.set_ticks(cbar_ticks)
		cbar.set_ticklabels([f"{tick:.1f}" for tick in cbar_ticks])
		plt.tight_layout()
		plt.axis('off')
		plt.savefig(outputFilename_LocRes[:-3]+"png", bbox_inches='tight', pad_inches=0.05) 

	return (preAddToName, float(actualRes_global_new), float(ratioSignal))


def main(mode, config, apix, odd_input, even_input, cpu_threads, gpu_enabled, gpu_settings, run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir, inputDir):
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)

	start_total = datetime.datetime.now()

	# Some configurations for the different input types
	if config == "Refined-Maps":
		runOnAveragedMap = True
		collapseWindow_i = False
		window_size_i = 0	
		test2 = False	
		p_cutoff = 0.01
	if config == "Micrographs":
		runOnAveragedMap = False
		collapseWindow_i = False 
		window_size_i = 0
		test2 = True
		p_cutoff = 0.05
	if config == "Tilt-Series":
		runOnAveragedMap = False
		collapseWindow_i = True
		window_size_i = 0
		test2 = True
		p_cutoff = 0.05
	if config == "Tomograms":
		runOnAveragedMap = False
		collapseWindow_i = False
		window_size_i = 0
		test2 = True
		p_cutoff = 0.05

	# Inputs
	resMax = 10
	accuracy_steps = 1
	referenceDistSize = 10000
	printDebugging = False
	boxValue = "max"
	spacingFilter = 0.05
	falloff = 1.5
	if run_fast:
		accuracy_steps = 2


   
	# Parse input GPUs			
	inputGPUs = []
	if gpu_settings != "Disabled" and len(gpu_settings) != 0:
		inputGPUs = list(np.array(gpu_settings.split(",")).astype(int))
   
	# Shared args tuple that gets passed to process_one_file
	shared_args = (mode, config, apix, None, None, cpu_threads, gpu_enabled, inputGPUs,
				   run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir, inputDir,
				   runOnAveragedMap, collapseWindow_i, window_size_i, test2, p_cutoff,
				   resMax, accuracy_steps, referenceDistSize, printDebugging, boxValue,
				   spacingFilter, falloff)

	# Handle single mode processing
	if mode != "batch":
		outputFilename_LocRes = os.path.join(outputDir, os.path.basename(odd_input)[:-4] + "_" + config + "_locRes" + ".mrc")
		if os.path.exists(outputFilename_LocRes):
			print("Warning: " + outputFilename_LocRes + " already exists. This file is processed already. For reprocessing, please delete output file or define new output directory. SKIP!\n")
			return     
		args = list(shared_args)
		args[3] = odd_input   # odd_input
		args[4] = even_input  # even_input
		process_one_file(tuple(args))
		print("IN TOTAL: " + str(datetime.datetime.now()-start_total) + "\n\n")
		return

	# Batch mode__________
 
	# handle multi-threading
	if gpu_enabled:
		global_threats = 1
		cpuThreads_batch = cpu_threads
		# if len(gpu_settings) != 0:
		# 	if len(inputGPUs) > 1:
		# 		global_threats = len(inputGPUs)
		# 		cpuThreads_batch = np.max([1, int(cpuThreads_batch//len(inputGPUs))])
	else:
		minThreads = 4
		global_threats = 1
		if cpu_threads > minThreads:
			global_threats = int(cpu_threads//minThreads)
			cpuThreads_batch = minThreads
     
	# handle input
	odd_id = odd_input
	even_id = even_input
	matching_files = [f for f in os.listdir(inputDir) if odd_id in f]
	it_loops = len(matching_files)

	# Build list of args for each file
	all_args = []
	for iterate_files in range(it_loops):
		odd_file = matching_files[iterate_files]
		even_file = odd_file.replace(odd_id, even_id)
		odd_path = os.path.join(inputDir, odd_file)

		outputFilename_LocRes = os.path.join(outputDir, os.path.basename(odd_path)[:-4] + "_" + config + "_locRes" + ".mrc")
		if os.path.exists(outputFilename_LocRes):
			print("Warning: " + outputFilename_LocRes + " already exists. This file is processed already. For reprocessing, please delete output file or define new output directory. SKIP!\n")
			continue

		even_path = os.path.join(inputDir, even_file)
		args = list(shared_args)
		args[3] = odd_path
		args[4] = even_path
		args[5] = cpuThreads_batch
		if global_threats > 1:
			if (gpu_settings != "Disabled") and (len(gpu_settings) != 0) and (global_threats != 0):
				args[7] = [inputGPUs[iterate_files%global_threats]]
			else:
				args[7] = []
		all_args.append(tuple(args))

	nFiles = len(all_args)
	print(f"Batch mode: found {nFiles} files to process")



	# Process files, each in a fresh worker
	nameArray, resGlobArray, ratioSignalArray = [], [], []
	result_queue = mp.Queue()
	completed = 0

	def _worker_wrapper(args, queue):
		result = process_one_file(args)
		queue.put(result)

	n_parallel = global_threats
	active_procs = []


	start_processing = datetime.datetime.now()
	# Start multiple threads
	for i, args in enumerate(all_args):
		proc = mp.Process(target=_worker_wrapper, args=(args, result_queue))
		proc.start()
		active_procs.append(proc)

		# Until limit hit
		if len(active_procs) >= n_parallel:
			for p in active_procs:
				p.join()
			while not result_queue.empty():
				result = result_queue.get()
				if result is not None:
					nameArray.append(result[0])
					resGlobArray.append(result[1])
					ratioSignalArray.append(result[2])
			completed += len(active_procs)
			print(f"Completed {completed}/{it_loops}")
			active_procs = []
			end_batch = datetime.datetime.now()
			current_time = end_batch-start_processing
			print("Estimated remaining time: " + str((current_time/completed)*(nFiles-completed)) + "\n\n")

	# Collect any remaining
	for p in active_procs:
		p.join()
	while not result_queue.empty():
		result = result_queue.get()
		if result is not None:
			nameArray.append(result[0])
			resGlobArray.append(result[1])
			ratioSignalArray.append(result[2])
	completed += len(active_procs)
	print(f"Completed {completed}/{it_loops}")
  
	# Write summary.tsv
	if len(nameArray) > 0:
		with open(os.path.join(outputDir, "summary.tsv"), 'w', encoding='utf-8') as file:
			file.write('id\tmedian_resolution\tsignal_ratio\n')
			nameArray = np.array(nameArray)
			resGlobArray = np.array(resGlobArray)
			ratioSignalArray = np.array(ratioSignalArray)
			sortedIndices = np.argsort(resGlobArray)
			nameArray = nameArray[sortedIndices]
			resGlobArray = resGlobArray[sortedIndices]
			ratioSignalArray = ratioSignalArray[sortedIndices]
			for i in range(len(nameArray)):
				file.write(f'{nameArray[i]}\t{resGlobArray[i]}\t{ratioSignalArray[i]}\n')

	print(f"\nBatch complete. Processed {len(nameArray)} files successfully.")
	print("IN TOTAL: " + str(datetime.datetime.now()-start_total) + "\n\n")

if __name__ == '__main__':    
	main()

