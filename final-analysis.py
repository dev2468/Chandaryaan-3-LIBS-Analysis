reference_database = {
    "Al": [281.65],
    "Fe": [241.88, 254.38, 502.87, 565.55, 757.64, 765.52],
    "Ti": [264.58],
    "S": [500.95, 555.88],
    "Cr": [762.26]
}
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import pywt


# Function to combine close wavelength values
def aggregate_close_wavelengths(wavelengths, counts, tolerance=0.1):
    aggregated_wavelengths = []
    aggregated_counts = []

    current_wavelength_group = [wavelengths[0]]
    current_counts_group = [counts[0]]

    for i in range(1, len(wavelengths)):
        if wavelengths[i] - current_wavelength_group[-1] <= tolerance:
            current_wavelength_group.append(wavelengths[i])
            current_counts_group.append(counts[i])
        else:
            # Average the close wavelengths and their counts
            aggregated_wavelengths.append(np.mean(current_wavelength_group))
            aggregated_counts.append(np.mean(current_counts_group))
            current_wavelength_group = [wavelengths[i]]
            current_counts_group = [counts[i]]

    # Add the final group
    if current_wavelength_group:
        aggregated_wavelengths.append(np.mean(current_wavelength_group))
        aggregated_counts.append(np.mean(current_counts_group))

    return np.array(aggregated_wavelengths), np.array(aggregated_counts)


# Smoothing
def smooth_data(spectrum, method='savgol', **kwargs):
    if method == 'savgol':
        spectrum_size = len(spectrum)
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 2)

        # Adjust window_length to be valid
        if window_length > spectrum_size:
            window_length = spectrum_size if spectrum_size % 2 == 1 else spectrum_size - 1  # Make it odd and <= spectrum_size

        # Ensure polyorder < window_length
        if polyorder >= window_length:
            polyorder = window_length - 1

        return savgol_filter(spectrum, window_length, polyorder)
    elif method == 'adjacent_averaging':
        return uniform_filter1d(spectrum, size=kwargs.get('window_size', 5))
    elif method == 'wavelet':
        coeffs = pywt.wavedec(spectrum, 'db1', mode='smooth')
        threshold = kwargs.get('threshold', 0.05)
        coeffs[1:] = [pywt.threshold(i, threshold * max(i)) for i in coeffs[1:]]
        return pywt.waverec(coeffs, 'db1', mode='smooth')
    else:
        raise ValueError("Unsupported smoothing method")


# Baseline correction
def baseline_correction(wavelengths, spectrum, method='polynomial', **kwargs):
    if method == 'polynomial':
        max_degree = len(wavelengths) - 1
        degree = min(kwargs.get('degree', 5), max_degree)  # Ensure degree is valid

        def polynomial(x, *coeffs):
            return sum(c * x ** i for i, c in enumerate(coeffs))

        try:
            coeffs, _ = curve_fit(polynomial, wavelengths, spectrum, p0=[0] * (degree + 1))
            baseline = polynomial(wavelengths, *coeffs)
        except RuntimeError:
            # Fallback: Use a simple linear fit if curve fitting fails
            z = np.polyfit(wavelengths, spectrum, 1)
            baseline = np.polyval(z, wavelengths)
    elif method == 'airPLS':
        baseline = airPLS(spectrum, lambda_=kwargs.get('lambda_', 10), p=kwargs.get('p', 0.01))
    else:
        raise ValueError("Unsupported baseline correction method")

    return spectrum - baseline


# Normalize
def normalize_data(spectrum, method='local_max'):
    if method == 'local_max':
        return spectrum / np.max(spectrum)
    elif method == 'area':
        return spectrum / np.trapz(spectrum)
    else:
        raise ValueError("Unsupported normalization method")


# Find peaks
def find_peaks_in_spectrum(spectrum, prominence_percentile=90):
    prominence = np.percentile(spectrum, prominence_percentile)
    peaks, _ = find_peaks(spectrum, prominence=prominence)
    return peaks


# Match peaks
def match_peaks_to_elements(peaks, wavelengths, reference_db, tolerance=0.9):
    matched_elements = []
    for peak in peaks:
        peak_wavelength = wavelengths[peak]
        matched = None
        for element, known_wavelengths in reference_db.items():
            for known_wavelength in known_wavelengths:
                if abs(peak_wavelength - known_wavelength) <= tolerance:
                    matched = element
                    print(matched, f"peak detected {peak_wavelength} | actual value in ISRO graph {known_wavelength}")
                    break
            if matched:
                break
        matched_elements.append((peak_wavelength, matched))
    return matched_elements


# Plot spectrum
def plot_spectrum(wavelengths, spectrum, matched_elements):
    plt.figure(figsize=(12, 8))
    plt.plot(wavelengths, spectrum, label="Processed Spectrum", color="blue")

    # Extract only matched peaks
    matched_wavelengths = [peak_wavelength for peak_wavelength, element in matched_elements if element]
    matched_intensities = [spectrum[np.abs(wavelengths - peak_wavelength).argmin()] for peak_wavelength in
                           matched_wavelengths]

    # Plot matched peaks
    plt.scatter(matched_wavelengths, matched_intensities, color="red", label="Matched Peaks", s=15)

    # Annotate matched peaks with element labels
    for peak_wavelength, element in matched_elements:
        if element:
            plt.annotate(f'{element} ({peak_wavelength:.2f} nm)',
                         (peak_wavelength, spectrum[np.abs(wavelengths - peak_wavelength).argmin()]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.title("Spectrum with Identified Peaks")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (Normalized Counts)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Main function
def process_reference_database(json_file_path, reference_db, wavelength_range=(200, 800),
                               smoothing_method='savgol', baseline_method='polynomial',
                               normalization_method='local_max', prominence_percentile=90, tolerance=0.5):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Convert JSON data to arrays, averaging intensities for the same wavelength
    wavelengths = []
    intensities = []
    for wavelength, intensity_list in data.items():
        wavelength_float = float(wavelength)
        wavelengths.append(wavelength_float)
        intensities.append(np.mean(intensity_list))  # Use mean intensity

    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    # Ensure there are no negative intensities
    intensities[intensities < 0] = 0

    # Filter within range
    valid_indices = (wavelengths >= wavelength_range[0]) & (wavelengths <= wavelength_range[1])
    wavelengths_filtered = wavelengths[valid_indices]
    spectrum_filtered = intensities[valid_indices]

    # Process spectrum
    spectrum_smoothed = smooth_data(spectrum_filtered, method=smoothing_method)
    spectrum_smoothed[spectrum_smoothed < 0] = 0  # Ensure no negative values after smoothing
    spectrum_corrected = baseline_correction(wavelengths_filtered, spectrum_smoothed, method=baseline_method)
    spectrum_corrected[spectrum_corrected < 0] = 0  # Ensure no negative values after baseline correction
    spectrum_normalized = normalize_data(spectrum_corrected, method=normalization_method)

    # Identify peaks
    peaks = find_peaks_in_spectrum(spectrum_normalized, prominence_percentile)

    # Match peaks to the reference database
    matched_elements = match_peaks_to_elements(peaks, wavelengths_filtered, reference_db, tolerance)

    # Plot the results
    plot_spectrum(wavelengths_filtered, spectrum_normalized, matched_elements)



# Execute the function
if __name__ == "__main__":
    json_file_path = "final_data_sorted.json"  # Replace with your JSON file path
    process_reference_database(json_file_path, reference_database)




json_file_path = "final_data_sorted.json"

process_libs_data(json_file_path, element_db, prominence=0.02, width=2, wavelength_range=(200, 800), tolerance=0.5,
                  smoothing_window=5, smoothing_polyorder=2)
