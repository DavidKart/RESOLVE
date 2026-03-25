"""Handling multi-threading, GPU, input parsing, writing"""

import mrcfile
import numpy as np
import os
import multiprocessing as mp
import datetime
import gc
from . import core, utils, utils_misc
import torch

def process_one_file(args):
	"""Process a single file pair in a worker process. Returns (name, resolution, signal_ratio) or None."""
	(mode, config, apix, odd_input, even_input, cpu_threads, gpu_enabled, gpu_settings,
	 run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir,
	 test2, p_cutoff, resMax, accuracy_steps, referenceDistSize,
	 spacingFilter, falloff) = args

	# Configurations for running on GPU
	gpu_ids = None
	if gpu_enabled:
		gpu_ids = gpu_settings
  
	if torch.cuda.is_available() and gpu_ids is not None:
		device = torch.device(f"cuda:{gpu_ids[0]}")
		print("GPU processing")
	else:
		device = torch.device("cpu")
		print("CPU processing")


    

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
	if mode != "batch":
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
		if not os.path.exists(signal_mask_input):
			print("Signal mask not found. Exit.")
			return None
		if len(signal_mask_input) == 0:
			signal_mask = None
		else:
			signal_mask = torch.from_numpy(mrcfile.open(signal_mask_input).data * 1).float().to(device)
			if signal_mask.shape != halfMap1Data.shape:
				print("Signal mask has not shape of input map. Exit.")
				return None
			signal_mask[signal_mask < 1] = 0
			signal_mask[signal_mask >= 1] = 1
			print("using signal mask for median estimate: " + str(signal_mask_input))
	if mask_strategy == "full_map":
		signal_mask = torch.ones(halfMap1Data.shape).to(device)

	# Reading pixel size
	if mode != "batch": print("Input configurations_____________")
	if apix is None:
		apix = np.round(float((halfMap1.voxel_size).x),2)
		apix_y = np.round(float((halfMap1.voxel_size).y),2)
		if dimension == 2:
			if mode != "batch": print("pixel size read from header (x,y): " + str(apix) + " " + str(apix_y))
		if dimension == 3:
			apix_z = np.round(float((halfMap1.voxel_size).z),2)
			if mode != "batch": print("pixel size read from header (x,y,z): " + str(apix) + " " + str(apix_y) + " " + str(apix_z)) # Z-value may differ for Tilt-series
	else:
		apix = float(apix)
	lowRes = resMax*apix # Lowest resolution 
	lowResMax = 1/(np.fft.rfftfreq(np.min(sizeMap))[1]/apix)
	lowRes = np.min([lowRes, lowResMax])
	if mode != "batch": print("lowest resolution to consider (10*apix): " + str(np.round(lowRes,2)))

	# This is for tilt-series 
	if config == "Tilt-Series":
		dimension_windows = 2
		stepSize = [5,5,5]
		stepSize[0] = 1 # Collapse z to 1. Not that for numpy arrays, x and z are swapped (z,y,x instead of x,y,z)
	else:
		dimension_windows = dimension
	if mode != "batch": print("using step size: " + str(" ".join(np.array(stepSize[::-1]).astype(str)))) # Adjust x-z swap

	# Get windows (radii) and shells
	sizeVol = 100
	resolutions = utils_misc.calculate_resolutions(sizeVol, apix, lowRes, accuracy_steps)

	# Get windows and box sizes from precalculated empirical simulations. Print out parameters used.
	windows = utils_misc.get_windows_empirical(apix, resolutions, dimension_windows)
	if mode != "batch": 
		print("using window radii [pix]: " + " ".join(map(str,np.round(windows,1))))
		print("to measure resolutions [Å]: " + " ".join(map(str,np.round(np.array(resolutions),2))))
		print("")

	# How many random maps are needed to get a good enough reference distribution
	windowtest = int(np.ceil(np.max(windows)))*2+1
	maxEntries = np.prod([windowtest for _ in range(dimension)])	
	if config == "Tilt-Series":
		maxEntries = np.prod([windowtest for _ in range(dimension_windows)])	
		possibleTests_nonOverlapping = int(np.prod(np.array(sizeMap[1:])+windowtest)/maxEntries)
		possibleTests_nonOverlapping = possibleTests_nonOverlapping**2 # Consider enhanced possibility space, dependencies are introduced in Fourier space, and are thus real-space location independent 
	else:
		possibleTests_nonOverlapping = int(np.prod(np.array(sizeMap)+windowtest)/maxEntries)
		possibleTests_nonOverlapping = possibleTests_nonOverlapping**2 # Consider enhanced possibility space, dependencies are introduced in Fourier space, and are thus real-space location independent 
	it_randomMaps = int(np.ceil(referenceDistSize / possibleTests_nonOverlapping))




	if config == "Tilt-Series":
		batchHalf1 = torch.from_numpy(halfMap1Data.astype(np.float32)).to(device)
		batchHalf2 = torch.from_numpy(halfMap2Data.astype(np.float32)).to(device)
	else:
		batchHalf1 = [torch.from_numpy(halfMap1Data.astype(np.float32)).to(device)]
		batchHalf2 = [torch.from_numpy(halfMap2Data.astype(np.float32)).to(device)]


	phasePermutation = True
	if config == "Refined-Maps":
		phasePermutation = False

	# Use batch processing here only for Tilt-series! 
	locResMap = core.compute_resolution(
		apix,
		list(windows),
		list(resolutions),
		batchHalf1,
		batchHalf2,
		stepSize[-1],
		gpu_ids,
		it_randomMaps,
		referenceDistSize,
		phasePermutation,
		4096,
		spacingFilter,
		falloff)
 
	if config != "Tilt-Series":
		locResMap = locResMap[0]
	else:
		locResMap = locResMap.permute(1, 0, 2, 3)

	# Naming file
	preAddToName = odd_input.split("/")[-1][:-4] + "_" + config + "_locRes"  
	outputFilename_LocRes = os.path.join(outputDir, preAddToName + ".mrc")
	pValueMapShape = locResMap[0].shape

	# cleanup 1
	del batchHalf1, batchHalf2
	torch.cuda.empty_cache()
	gc.collect()
 
	localResMap_out = utils_misc.fill_map(locResMap, torch.tensor(resolutions, device=device, dtype = locResMap[0].dtype), p_cutoff, lowRes, test2)
	localResMap_out[localResMap_out>lowRes] = lowRes   

	# cleanup 2
	locResMap = locResMap.cpu().numpy()
	torch.cuda.empty_cache()
 
	# median p-value creation
	actualRes_global_new, signalRatio = None, None
	if config != "Refined-Maps":
		if signal_mask == None:
			signalMask_stepSize = torch.ones(pValueMapShape)
			signalMask_stepSize[localResMap_out >= lowRes] = 0
		else:
			if dimension == 2:
				signalMask_stepSize = signal_mask[::stepSize[0], ::stepSize[1]]   
			if dimension == 3:
				signalMask_stepSize = signal_mask[::stepSize[0], ::stepSize[1], ::stepSize[2]]   				
		dict_tilt_series, signalRatio = utils_misc.calculate_median_res(locResMap, signalMask_stepSize, resolutions, dimension, config, mask_measure)
		actualRes_global_new = utils_misc.write_medianRes(dict_tilt_series, signalRatio, resolutions, p_cutoff, lowRes, config, mode, outputDir, preAddToName, mask_measure)

	# cleanup 3
	del locResMap
	if signal_mask is not None:
		del signal_mask
	if config != "Refined-Maps":
		del signalMask_stepSize
	torch.cuda.empty_cache()
	gc.collect()

	# Interpolating
	# The grid needs to be interpolated in the end.
	gc.collect() # Make sure garbage is collected before interpolating to free memory
	print("interpolating grid")
	if np.max(stepSize) != 1: # This is always the case in this default script
		if config == "Tilt-Series": # For tilt-series
			localResMap = []
			for i in range(localResMap_out.shape[0]):
				localResMap.append(utils_misc.interpolate_with_zoom(localResMap_out[i], sizeMap[1:], stepSize, lowRes))
			localResMap = torch.stack(localResMap)
		else:
			localResMap = utils_misc.interpolate_with_zoom(localResMap_out, sizeMap, stepSize, lowRes)
	else:
		localResMap = np.copy(localResMap_out)    
	# del localResMap_out 
	localResMap[torch.isnan(localResMap)] = lowRes

	localResMap = localResMap.cpu().numpy()


	# Write output
	print("write here: " + str(outputFilename_LocRes))
	localResMapMRC = mrcfile.new(outputFilename_LocRes, overwrite=True)
	localResMap = np.float32(localResMap)
	localResMapMRC.set_data(localResMap)
	if config == "Tilt-Series":
		localResMapMRC.voxel_size = (apix,apix,apix)
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

	return preAddToName, actualRes_global_new, signalRatio
 


def main(mode, config, apix, odd_input, even_input, cpu_threads, gpu_enabled, gpu_settings, run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir, inputDir):
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)

	start_total = datetime.datetime.now()

	# Some configurations for the different input types
	if config == "Refined-Maps":
		test2 = False	
		p_cutoff = 0.01
	if config == "Micrographs":
		test2 = True
		p_cutoff = 0.05
	if config == "Tilt-Series":
		test2 = True
		p_cutoff = 0.05
	if config == "Tomograms":
		test2 = True
		p_cutoff = 0.05

	# Inputs
	resMax = 10
	accuracy_steps = 1
	referenceDistSize = 10000
	spacingFilter = 0.05
	falloff = 1.5
	if run_fast:
		accuracy_steps = 2


   
	# Parse input GPUs			
	inputGPUs = []
	if gpu_settings != "Disabled":
		if len(gpu_settings) != 0: inputGPUs = list(np.array(gpu_settings.split(",")).astype(int))
		if len(gpu_settings) == 0: inputGPUs = list(range(torch.cuda.device_count()))

	# Shared args tuple that gets passed to process_one_file
	shared_args = (mode, config, apix, None, None, cpu_threads, gpu_enabled, inputGPUs,
				   run_fast, mask_strategy, signal_mask_input, mask_measure, outputDir,
				   test2, p_cutoff, resMax, accuracy_steps, referenceDistSize,
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

	# ____________Batch mode__________
 
	
	# handle multi-threading
	if gpu_enabled:
		global_threats = 1
		cpuThreads_batch = cpu_threads
		if len(inputGPUs) > 1:
			global_threats = len(inputGPUs)
			cpuThreads_batch = np.max([1, int(cpuThreads_batch//len(inputGPUs))])
	else:
		minThreads = 1
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
			if (gpu_settings != "Disabled") and (global_threats != 0):
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
	if len(nameArray) > 0 and config != "Refined-Maps":
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

