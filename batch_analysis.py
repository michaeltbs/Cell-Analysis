import os
import argparse
import pandas as pd
from src.workflow import run_workflow, save_image_with_boundaries

def batch_process_images(input_dir, output_dir, age, use_channels=['gal'], region='lower', threshold_method='otsu', window_size=41, disk_radius=2, min_distance=7):
    """
    Process all .czi images in a directory, aggregate results, and save as a CSV.

    Args:
        input_dir (str): Path to the directory containing .czi files.
        output_dir (str): Path to the directory where results will be saved.
        age (str): The age to be added as a column to the final CSV.
        use_channels (list, optional): Channels to use for analysis. Defaults to ['gal'].
        region (str, optional): Region to analyze. Defaults to 'lower'.
        threshold_method (str, optional): Thresholding method. Defaults to 'otsu'.
        window_size (int, optional): Window size for local thresholding. Defaults to 41.
        disk_radius (int, optional): Disk radius for morphological operations. Defaults to 2.
        min_distance (int, optional): Minimum distance for watershed segmentation. Defaults to 7.
    """
    # Find all .czi files in the input directory
    czi_files = [f for f in os.listdir(input_dir) if f.endswith('.czi')]
    
    if not czi_files:
        print(f"No .czi files found in {input_dir}")
        return

    all_measurements = []

    for filename in czi_files:
        file_path = os.path.join(input_dir, filename)
        print(f"Processing {file_path}...")

        # Run the analysis workflow
        results = run_workflow(
            file_path=file_path,
            use_channels=use_channels,
            region=region,
            threshold_method=threshold_method,
            window_size=window_size,
            disk_radius=disk_radius,
            min_distance=min_distance
        )

        if results and 'measurements' in results:
            measurements = results['measurements']
            
            # Add identifiers to the measurements DataFrame
            measurements['filename'] = filename
            measurements['age'] = age
            measurements['channel'] = ','.join(use_channels)
            measurements['region'] = region
            
            all_measurements.append(measurements)
            
            # Save individual image results
            input_basename = os.path.splitext(filename)[0]
            # Create a unique output directory for each image based on channel and region
            channel_str = '_'.join(use_channels)
            output_subdir = os.path.join(output_dir, f"{channel_str}_{region}_{input_basename}")
            
            save_image_with_boundaries(
                results=results,
                output_dir=output_subdir,
                filename_prefix=input_basename
            )

    if all_measurements:
        # Concatenate all measurements into a single DataFrame
        final_df = pd.concat(all_measurements, ignore_index=True)
        
        # Save the aggregated DataFrame to a CSV file
        output_csv_path = os.path.join(output_dir, f"{age}_{region}_batch_analysis_results.csv")
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nBatch processing complete. Aggregated results saved to {output_csv_path}")
    else:
        print("No measurements were generated during batch processing.")

if __name__ == "__main__":
    
    
    batch_process_images("/Users/Uni/Desktop/Coding/project_cellanalysis/data/raw_czi/Input_Old_V2", 
                         "/Users/Uni/Desktop/Coding/project_cellanalysis/outputs/old/gal", "old", region='upper', use_channels=['gal'])
