# colour_system.py
import numpy as np
import yaml
from scipy.interpolate import RectBivariateSpline

class RefractiveIndex:
    """
    A class to load, parse, and interpolate refractive index data from a YAML file.
    """
    def __init__(self, file_path, wavelength_min, wavelength_max, spectral_bands):
        """
        Initializes the RefractiveIndex object.

        Args:
            file_path (str): The path to the .yml file.
            target_wavelengths_nm (np.ndarray): A numpy array of wavelengths (in nm)
                                                 for which to interpolate n and k.
        """
        self.file_path = file_path
        self.target_wavelengths_nm = np.linspace(wavelength_min, wavelength_max, spectral_bands)
        #self.target_wavelengths_nm = target_wavelengths_nm
        self.n, self.k = self._load_and_interpolate_data()

    def _load_and_interpolate_data(self):
        """
        Loads refractive index data from a refractiveindex.info YAML file,
        and interpolates it to a target set of wavelengths.

        Returns:
            tuple[list, list]: A tuple containing two lists:
                               - The interpolated refractive index (n) values.
                               - The interpolated extinction coefficient (k) values.
        """
        try:
            with open(self.file_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return None, None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None, None

        # Extract the tabulated data string
        raw_data_string = ""
        for item in data.get('DATA', []):
            if item.get('type') == 'tabulated nk':
                raw_data_string = item.get('data')
                break

        if not raw_data_string:
            print(f"Error: Could not find 'tabulated nk' data in {self.file_path}.")
            return None, None

        # Parse the raw data string into lists
        source_wavelengths_um = []
        source_n = []
        source_k = []
        for line in raw_data_string.strip().split('\n'):
            parts = line.split()
            if len(parts) == 3:
                try:
                    # Data is in: wavelength (μm), n, k
                    source_wavelengths_um.append(float(parts[0]))
                    source_n.append(float(parts[1]))
                    source_k.append(float(parts[2]))
                except ValueError:
                    print(f"Warning: Could not parse line in YAML data: '{line}'")
                    continue

        # Convert source wavelengths from micrometers (μm) to nanometers (nm)
        source_wavelengths_nm = [wl * 1000 for wl in source_wavelengths_um]

        # Use numpy to interpolate the n and k values at the target wavelengths
        sort_indices = np.argsort(source_wavelengths_nm)
        source_wavelengths_nm = np.array(source_wavelengths_nm)[sort_indices]
        source_n = np.array(source_n)[sort_indices]
        source_k = np.array(source_k)[sort_indices]

        interpolated_n = np.interp(self.target_wavelengths_nm, source_wavelengths_nm, source_n)
        interpolated_k = np.interp(self.target_wavelengths_nm, source_wavelengths_nm, source_k)

        return interpolated_n.tolist(), interpolated_k.tolist()


class ColourSystem:
    def __init__(self, file_path, red, green, blue, white, wavelength_min, wavelength_max, spectral_bands):
 
        # Chromaticities
        self.red = self.xyz_from_xy(red)
        self.green = self.xyz_from_xy(green)
        self.blue = self.xyz_from_xy(blue)
        self.white = self.xyz_from_xy(white)
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

        # CIE color matching functions
        self.cmf_raw = np.loadtxt(file_path, comments='#')
        self.target_wavelengths = np.linspace(wavelength_min, wavelength_max, spectral_bands)

        self.cmf = self._interpolate_cmf()
        self.wavelength_step = (wavelength_max - wavelength_min) / (spectral_bands - 1)
        self.normalization_factor = np.sum(self.cmf[:, 1]) * self.wavelength_step
    
    def xyz_from_xy(self, vec):
        """Return the vector (x, y, 1-x-y)."""
        x = vec[0]
        y = vec[1]
        return np.array((x, y, 1-x-y))

    def _interpolate_cmf(self):
        """
        Interpolates the CIE CMF to the target wavelengths.
        """
        source_wavelengths = self.cmf_raw[:, 0]
        source_xyz = self.cmf_raw[:, 1:]

        # Interpolate each component (X, Y, Z)
        interp_x = np.interp(self.target_wavelengths, source_wavelengths, source_xyz[:, 0])
        interp_y = np.interp(self.target_wavelengths, source_wavelengths, source_xyz[:, 1])
        interp_z = np.interp(self.target_wavelengths, source_wavelengths, source_xyz[:, 2])

        return np.vstack((interp_x, interp_y, interp_z)).T

class SpecularVec:
    def __init__(self, file_path, scale, wavelength_min, wavelength_max, spectral_bands):
        self.raw = np.loadtxt(file_path, comments='#')
        self.target_wavelengths = np.linspace(wavelength_min, wavelength_max, spectral_bands)
        self.vec = self._interpolate(scale)

    def _interpolate(self,scale):
        source_wavelengths = self.raw[:, 0]
        source_values = self.raw[:, 1]

        # Interpolate each component (X, Y, Z)
        interp = np.interp(self.target_wavelengths, source_wavelengths, source_values) * scale

        return interp.tolist()

class EEMMatrix:
    _next_eem_id = 1
    def __init__(self, wavelength_min, wavelength_max, spectral_bands, file_path=None, matrix=None):
        self.target_wavelengths = np.linspace(wavelength_min, wavelength_max, spectral_bands)
        if file_path:
            self.matrix = self._load_and_interpolate(file_path)
        elif matrix:
            self.matrix = matrix
        else:
            raise ValueError(
                "EEMMatrix Missing file_path and Matrix, one of either must be provided"
            )

        self.id = EEMMatrix._next_eem_id
        EEMMatrix._next_eem_id += 1

    def _load_and_interpolate(self, file_path):
        # Load the raw text file 
        raw_data = np.loadtxt(file_path, comments='#')
        
        # Extract axes and values
        source_emissions = raw_data[0, 1:]
        source_excitations = raw_data[1:, 0]
        z_values = raw_data[1:, 1:]
        
        # reate a 2D interpolator
        interpolator = RectBivariateSpline(source_excitations, source_emissions, z_values)
        
        # Evaluate across your engine's target spectral bands
        interpolated_matrix = interpolator(self.target_wavelengths, self.target_wavelengths)
        
        # Enforce Stokes Shift (zero out where emission <= excitation)
        for i in range(len(self.target_wavelengths)):
            for j in range(len(self.target_wavelengths)):
                if self.target_wavelengths[j] <= self.target_wavelengths[i]:
                    interpolated_matrix[i, j] = 0.0

        return np.maximum(0, interpolated_matrix) # Prevent negative interpolation artifacts