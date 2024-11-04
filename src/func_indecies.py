import numpy as np

###############################  Vegitation Indecies ######################################

def nbr(event):
    # Calculate the components that make up the NBR calculation
    band_diff = event.B08 - event.B12
    band_sum = event.B08 + event.B12

    # Calculate NBR and store it as a measurement in the original dataset
    return  band_diff / band_sum

def ndvi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

def ndre(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B09 - event.B05
    band_sum = event.B09 + event.B05

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

def ndwi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B03 - event.B08
    band_sum = event.B03 + event.B08

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

# Wetness = 0.1509 (Band 1) + 0.1973 (Band 2) + 0.3279 (Band 3) + 0.3406 (Band 4) – 0.7112 (Band 5) – 0.4572 (Band 7)
def tcw(event):
    tcw = 0.1509 * event.B02 + 0.1973 * event.B03 + 0.3279 * event.B04 + 0.3406 * event.B08 - 0.7112 * event.B11 - 0.4572 * event.B12
    return tcw

# Greenness = – 0.2848 (Band 1) – 0.2435 (Band 2) – 0.5436 (Band 3) + 0.7243 (Band 4) + 0.0840 (Band 5) – 0.1800 (Band 7)
def tcg(event):
    tcg = -0.2848 * event.B02 - 0.2435 * event.B03 - 0.5436 * event.B04 + 0.7243 * event.B08 + 0.0840 * event.B11 - 0.1800 * event.B12
    return tcg

# Brightness = 0.3037 (Band 1) + 0.2793 (Band 2) + 0.4743 (Band 3) + 0.5585 (Band 4) + 0.5082 (Band 5) + 0.1863 (Band 7)
def tcb(event):
    tcb = 0.3037 * event.B02 + 0.2793 * event.B03 + 0.4743 * event.B04 + 0.5585 * event.B08 + 0.5082 * event.B11 + 0.1863 * event.B12
    return tcb

def drs(event):
   
    red_band_power = event.B04 ** 2
    nir_band_power = event.B12 ** 2
    band_sqrt = np.sqrt(red_band_power + nir_band_power)

    return band_sqrt

def ndrs(event):

    drs_values = event.drs

    # Calculate the minimum and maximum values
    min_value = np.min(drs_values)
    max_value = np.max(drs_values)

    # Normalize the values to the range [0, 1]
    normalized_values = (drs_values - min_value) / (max_value - min_value)

    return normalized_values

def ndmi(event):
    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B11
    band_sum = event.B08 + event.B11

    # Calculate NDVI and store it as a measurement in the original dataset
    return band_diff / band_sum

def nirv(event, C=0.08):

    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04
    
    p2 = event.B08

    # Calculate NDVI and store it as a measurement in the original dataset
    ndvi = band_diff / band_sum

    nirv = (ndvi - C)* p2

    return nirv
 
def kndvi(event, sigma):
    # Extract red and near-infrared band values
    red_band_value = event.B04
    nir_band_value = event.B12
    
    # Calculate the squared difference
    squared_difference = (nir_band_value - red_band_value)
    
    # Calculate the divisor
    divisor = (2 * sigma)
    
    # Calculate the expression inside tanh
    expression = (squared_difference / divisor)**2
    
    # Calculate and return the kndvi using the hyperbolic tangent
    kndvi = np.tanh(expression)
    
    return kndvi

def kndvi05(event):

    # Calculate the components that make up the NDVI calculation
    band_diff = event.B08 - event.B04
    band_sum = event.B08 + event.B04

    mid = band_diff / band_sum

    tan = np.tanh(mid ** 2) 
    
    return tan

def k(r, s, sigma):
    return np.exp(-((r - s) ** 2) / (2 * sigma ** 2))

def kdrs(event, sigma):
    red = event.B04
    swir = event.B12

    return 2 * (1 + k(red, swir, sigma))

def kndrs(event):

    kDRS_values = event.kdrs

    # Calculate the minimum and maximum values
    min_value = np.min(kDRS_values)
    max_value = np.max(kDRS_values)

    # Normalize the values to the range [0, 1]
    normalized_values = (kDRS_values - min_value) / (max_value - min_value)

    return normalized_values
