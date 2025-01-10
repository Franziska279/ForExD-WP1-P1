#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:05:37 2024

@author: ebutler
"""
import argparse
import cdsapi


def main(idx):

    c = cdsapi.Client()

    c.retrieve(
        'cems-fire-historical-v1',
        {
            'product_type': 'reanalysis',
            'variable': 'fire_weather_index',
            'dataset_type': 'consolidated_dataset',
            'system_version': '4_1',
            'year': [
                {idx},
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'grid': '0.25/0.25',
            'format': 'netcdf',
        },
        '/Net/Groups/BGI/work_2/ForExD/WP2/Data/FireWeather/FireWeatherIndex_2010-2019.nc')


if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Sentinel-2 Data Downloader")
    parser.add_argument("index", type=int, help="Index for the grid to process (from SLURM_ARRAY_TASK_ID)")
    
    # Parse arguments
    args = parser.parse_args()
     
    main(metadata_output=args.index)
