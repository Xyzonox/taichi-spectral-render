# Taichi Spectral Render

This project is a physically-based, multi-spectral path tracing engine developed using Python and Taichi Lang. Unlike traditional RGB renderers, this engine operates in the spectral domain, simulating light transport across a discrete set of wavelengths to accurately capture complex optical phenomena like dispersion, fluorescence, and wavelength-dependent absorption.

### Key Features

- Hyperspectral Rendering: Simulates light across 25+ spectral bands (expandable) instead of simple RGB, allowing for realistic material interactions across the visible and non-visible spectrum.

- Complex Refractive Index ($n, k$): Handles materials with complex IOR (Index of Refraction and Extinction Coefficient), enabling accurate simulation of metals (like Gold) and semiconductors (like Silicon).

- Volumetric Path Tracing: Supports true volumetric effects, including:Henyey-Greenstein Scattering: For simulating anisotropic media like clouds, skin, or foggy environments.

- Spectral Absorption: Wavelength-dependent light attenuation within volumes.

- Fluorescence & Re-emission: Implements Excitation-Emission Matrices (EEM) to simulate materials that absorb light at one wavelength and re-emit it at another.

- GGX Specular Reflection/Transmission: Microfacet model for realistic roughness and highlights.

- Oren-Nayar Diffuse: for rough, matte surfaces that deviate from Lambertian behavior.

- Next Event Estimation (NEE): Direct light sampling to reduce variance and noise.

- Multiple Importance Sampling (MIS): Balances BSDF sampling and light sampling for efficient convergence.

- GPU Accelerated: Built on Taichi, utilizing CUDA for high-performance parallel processing on the GPU.

- BVH Acceleration: Uses a Bounding Volume Hierarchy to handle complex triangle meshes efficiently.

# Project Structure

- main.py: The core engine, containing Taichi kernels for ray-surface intersection, BVH traversal, and the path tracing integration loop.

- data.py: Utilities for loading and interpolating spectral data ($n, k$ values), Color Matching Functions (CMF), and coordinate systems.

- SpecEdit.html: An interactive webui used to generate CMFs, three arrays of values for an array of wavelengths (nm), and SpectralVecs, a singular array of values for an array o wavelengths (nm).

- data/CMF/: (Directory) Contains various color matching functions to convert spectral data to perceivable color.

- data/refraction_index: (Directory) Contains .yml for refractive indices and .txt files for CMF and spectral data.

- data/SpecVecs: (Directory) Contains miscellaneous SpecularVec files.

# Useage

Ensure you have Python 3.8+ installed. The following libraries are required:

- Taichi: The core high-performance computing framework.

- NumPy: For data manipulation and scene setup.

- PyYAML: To parse refractive index data and configuration files.

Install them via pip:

```
pip install taichi numpy pyyaml
```

### Scene Configuration
The scene is defined in the create_scene() function within main.py. You can modify the materials list and the meshes list to add geometry and define optical properties.

### Data Generation
Using SpecEdit.html, CMFs and SpecrtalVecs can be created visually by adjusting a bezier curve. When finished, the specrtal data can be exported and used in the renderer. SpecrtalVecs can be used to define unique material properties.

Instead of generating multiple SpectralVecs, spectral information from the refractiveidex.info database can be utilized. Simply locate the desired material, access its full database record, then download the YAML file. Note: only entries with tabulated spectral data are supported.

### Running the Renderer
Execute the main script:

```
python main.py
```
Once the window opens, you can interact with the renderer:

S Key: Toggle between Render Mode (Full spectral path tracing) and Geometry Mode (Visualizes surface normals).

C Key: Reset the accumulator (useful if you modify camera parameters in code).

ESC: Close the application.
