import argparse
import sys
from scripts.resolve import main as resolve_main
import os

def main():
    parser = argparse.ArgumentParser(description="Run resolve.py without GUI")
    
    parser.add_argument("--mode", choices=["single", "batch"], required=True, 
                        help="Execution mode: 'single' for file inputs, 'batch' for ID-based processing.")
    parser.add_argument("--config", choices=["Refined-Maps", "Micrographs", "Tilt-Series", "Tomograms"], 
                        required=True, help="Select the configuration type.")
    parser.add_argument("--apix", required=False, help="Pixel Size. If not given, reading header.")
    parser.add_argument("--outputDir", required=True, help="Output directory.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="Number of CPU threads (default: 4).")
    parser.add_argument("--gpu_enabled", action="store_true", help="Enable GPU usage. Not recommended for micrographs.")
    parser.add_argument("--gpu_settings", default="", help="GPU settings (only used if --gpu_enabled is set). List GPUs to use, comma separated. By default, all available GPUs are used.")
    parser.add_argument("--fast", action="store_true", help="Lower sampling in Fourier space and real space. Faster, needs less memory.")

    # Input mask & measurement mode
    parser.add_argument("--maskFile", default="", help="Focus global resolution estimates with an input mask (default: empty string).")
    parser.add_argument("--mask_measure", choices=["median", "average"], default="median", 
                        help="Measure to calculate global resolution from local measurements (default: median).")

    # Arguments for single-run mode
    parser.add_argument("--input1", help="Path to the first input file (required for 'single' mode).")
    parser.add_argument("--input2", help="Path to the second input file (required for 'single' mode).")

    # Arguments for batch mode
    parser.add_argument("--odd_id", help="ID for the odd dataset (required for 'batch' mode).")
    parser.add_argument("--even_id", help="ID for the even dataset (required for 'batch' mode).")
    parser.add_argument("--inputDir", help="Path to the input directory (required for 'batch' mode).")

    args = parser.parse_args()

    if not resolve_main:
        print("Error: Could not import resolve.py's main function!")
        sys.exit(1)

    if args.mode == "single":
        if not args.input1 or not args.input2:
            print("Error: --input1 and --input2 are required in 'single' mode.")
            sys.exit(1)
        odd_arg = args.input1
        even_arg = args.input2
        input_dir = None  
    else:  # Batch mode
        if not args.odd_id or not args.even_id:
            print("Error: --odd_id and --even_id are required in 'batch' mode.")
            sys.exit(1)
        if not args.inputDir:
            print("Error: --inputDir is required in 'batch' mode.")
            sys.exit(1)
        odd_arg = args.odd_id
        even_arg = args.even_id
        input_dir = args.inputDir.strip()
        if not input_dir:
            input_dir = os.getcwd()
        else:
            input_dir = os.path.abspath(input_dir)

    outputDir = args.outputDir.strip()
    if not outputDir:
        outputDir = os.getcwd()
    else:
        outputDir = os.path.abspath(outputDir)

    resolve_main(
        mode=args.mode,
        config=args.config,
        apix=args.apix,
        odd_input=odd_arg,
        even_input=even_arg,
        cpu_threads=args.cpu_threads,
        gpu_enabled=args.gpu_enabled,
        gpu_settings=args.gpu_settings if args.gpu_enabled else "Disabled",
        run_fast = args.fast,
        signal_mask_input=args.maskFile,
        mask_measure=args.mask_measure,
        outputDir=outputDir,
        inputDir=input_dir,
    )

if __name__ == "__main__":
    main()
