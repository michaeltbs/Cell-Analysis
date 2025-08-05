import os
import pandas as pd
import numpy as np
from src.workflow import run_workflow, save_image_with_boundaries

def batch_process_images(input_dir, output_dir, age, use_channels=['gal'], region='ARC',
                          threshold_method='otsu', window_size=41, disk_radius=2, min_distance=7):
    """
    Process all .czi images in a directory, aggregate results, and save as a CSV.
    Applies the same processing parameters (thresholding, morphological footprint) to all channel sets.
    """
    # Map display region to processing region
    processing_region = "upper" if region == "DMH" else "lower"

    if not os.path.isdir(input_dir):
        print(f"\n--- WARNUNG: Input-Verzeichnis nicht gefunden, wird übersprungen: {input_dir} ---\n")
        return None

    czi_files = [f for f in os.listdir(input_dir) if f.endswith('.czi')]
    if not czi_files:
        print(f"Keine .czi-Dateien in {input_dir} gefunden.")
        return None

    all_measurements = []
    for filename in czi_files:
        file_path = os.path.join(input_dir, filename)
        print(f"Verarbeite {file_path}...")

        # Run the complete workflow with provided parameters
        results = run_workflow(
            file_path=file_path,
            use_channels=use_channels,
            region=processing_region,
            threshold_method=threshold_method,
            window_size=window_size,
            disk_radius=disk_radius,
            min_distance=min_distance
        )

        # Collect results if available
        if results and 'measurements' in results and not results['measurements'].empty:
            measurements = results['measurements']
            measurements['filename'] = filename
            measurements['age'] = age
            measurements['channel'] = ','.join(use_channels)
            measurements['region'] = region
            measurements['total_image_int_den'] = np.sum(results.get('image_to_process', 0))
            all_measurements.append(measurements)

            # Save boundary overlay images
            basename = os.path.splitext(filename)[0]
            save_dir = os.path.join(output_dir, basename)
            save_image_with_boundaries(
                results=results,
                output_dir=save_dir,
                filename_prefix=basename
            )

    # Aggregate detailed data and save
    if all_measurements:
        final_df = pd.concat(all_measurements, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        chan_str = '_'.join(use_channels)
        csv_name = f"detailed_measurements_{age}_{region}_{chan_str}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        final_df.to_csv(csv_path, index=False)
        print(f"\nStapelverarbeitung für '{age} - {region}' abgeschlossen. Messergebnisse gespeichert in {csv_path}")
        return final_df
    else:
        print(f"Keine Messungen für '{age} - {region}' generiert.")
        return None

if __name__ == "__main__":
    current_path = os.getcwd()

    # Define channel combinations (single and multi-channel)
    channels_to_process = [
        ['gal'], ['pomc'], ['glp1r'],
        ['gal','pomc'], ['gal','glp1r'], ['pomc','glp1r'],
        ['gal','pomc','glp1r']
    ]

    # Define jobs (age groups and regions)
    all_jobs = [
        {"age":"Adult","folder_name":"Input_Adult","region":"ARC"},
        {"age":"Adult","folder_name":"Input_Adult","region":"DMH"},
        {"age":"Old","folder_name":"Input_Old","region":"ARC"},
        {"age":"Old","folder_name":"Input_Old","region":"DMH"},
    ]

    # Unified processing parameters for all channels
    processing_params = {
        'threshold_method': 'otsu',
        'window_size': 41,
        'disk_radius': 2,
        'min_distance': 7
    }

    # Root output directory
    output_root = os.path.join(current_path, 'Output')
    all_final_summaries = []

    for channel_group in channels_to_process:
        key = '_'.join(channel_group)
        print(f"\n=== ANALYSE KANAL: {key} ===")

        # If 'pomc' in channels, only process ARC region
        jobs = [job for job in all_jobs if not ('pomc' in channel_group and job['region'] != 'ARC')]

        for job in jobs:
            age = job['age']
            region = job['region']
            input_dir = os.path.join(current_path, 'data', 'raw_czi', job['folder_name'])
            output_dir = os.path.join(output_root, age, key, region)

            # Process batch
            detailed_df = batch_process_images(
                input_dir=input_dir,
                output_dir=output_dir,
                age=age,
                use_channels=channel_group,
                region=region,
                **processing_params
            )

            if detailed_df is not None:
                # Summarize per file
                summary_df = detailed_df.groupby(['filename', 'age', 'region', 'channel']).agg(
                    cell_count=('label', 'size'),
                    mean_area=('area', 'mean'),
                    mean_intensity=('mean_intensity', 'mean'),
                    mean_integrated_density=('integrated_density', 'mean'),
                    total_image_int_den=('total_image_int_den', 'first')
                ).reset_index()
                summary_df.rename(columns={
                    'mean_area': 'mean_area_per_cell',
                    'mean_intensity': 'mean_intensity_per_cell',
                    'mean_integrated_density': 'mean_integrated_density_per_cell'
                }, inplace=True)

                # Save
                os.makedirs(output_dir, exist_ok=True)
                summary_path = os.path.join(output_dir, f"final_summary_{age}_{region}_{key}.csv")
                summary_df.to_csv(summary_path, index=False)
                print(f"Finale Zusammenfassung {age}/{region}/{key} gespeichert unter: {summary_path}")

                all_final_summaries.append(summary_df.assign(age=age, region=region, channel=key))

    # Create consolidated summary CSV
    if all_final_summaries:
        big_df = pd.concat(all_final_summaries, ignore_index=True)
        big_csv_path = os.path.join(output_root, 'all_final_summaries.csv')
        big_df.to_csv(big_csv_path, index=False)
        print(f"\nGesamt-Zusammenfassung gespeichert unter: {big_csv_path}")

    print("\nAlle Analysen abgeschlossen.")
