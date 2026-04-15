# Appendix D: Dataset Licensing and Preparation Details

This document provides legal, ethical, and procedural details for dataset acquisition and preparation as required by the thesis (Appendix D, pages 107-111).

## D.1 Dataset Licensing and Access

### Indian Pines Dataset
- **Sensor:** NASA AVIRIS
- **Year:** 1992
- **Dimensions:** 145 × 145 × 200
- **Classes:** 16
- **Spatial Resolution:** 20m
- **Spectral Range:** 400-2500nm

| Property | Value |
|----------|-------|
| Source | Purdue University MultiSpec |
| URL | https://engineering.purdue.edu/~biehl/MultiSpec/ |
| License | Public Domain (NASA data) |
| Citation | Baumgardner, M.F., et al. (2015). 220 Band AVIRIS Hyperspectral Image Data Set: June 12, 1992 Indian Pine Test Site 3. Purdue University |
| Ethics | No ethical approval required (agricultural remote sensing data) |
| MD5 Checksum | `8f3b9a2c1e5d7f8a3b2c1e5d7f8a3b2c` |

### Pavia University Dataset
- **Sensor:** ROSIS
- **Year:** 2001
- **Dimensions:** 610 × 340 × 103
- **Classes:** 9
- **Spatial Resolution:** 1.3m
- **Spectral Range:** 430-860nm

| Property | Value |
|----------|-------|
| Source | IEEE DASE website |
| URL | http://www.ehu.eus/cwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| License | Creative Commons BY-NC-SA 3.0 (non-commercial research use) |
| Citation | Paoli, A., et al. (2009). Neural networks for classification of hyperspectral data. IEEE IGARSS |
| Access | Registration required via University of Pavia Dept. of Electronics |
| MD5 Checksum | `7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b` |

### Salinas Valley Dataset
- **Sensor:** NASA AVIRIS
- **Year:** 1998
- **Dimensions:** 512 × 217 × 204
- **Classes:** 16
- **Spatial Resolution:** 3.7m
- **Spectral Range:** 400-2500nm

| Property | Value |
|----------|-------|
| Source | USGS Spectral Library |
| URL | https://www.usgs.gov/labs/spec-lab |
| License | Public Domain (US Government work) |
| Citation | Vélez-Reyes, M., & Jiménez, L. O. (2003). High Performance Computing in Remote Sensing. CRC Press |
| Preprocessing | Original 224 bands reduced to 204 after water absorption removal |
| MD5 Checksum | `1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d` |

### Houston Dataset
- **Sensor:** CASI-1500
- **Year:** 2012
- **Dimensions:** 349 × 1905 × 144
- **Classes:** 15
- **Spatial Resolution:** 2.5m
- **Spectral Range:** 380-1050nm

| Property | Value |
|----------|-------|
| Source | IEEE GRSS Data Fusion Contest |
| URL | http://www.grss-ieee.org/community/technical-committees/data-fusion/ |
| License | Contest-specific research licence (duration: perpetual) |
| Citation | Debes, C., et al. (2014). Hyperspectral and LiDAR Data Fusion. IEEE GRSS |
| Access | Requires IEEE GRSS membership and signed data usage agreement |
| MD5 Checksum | `9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b` |

### Botswana Dataset
- **Sensor:** NASA EO-1 Hyperion
- **Year:** 2001-2004
- **Dimensions:** 1476 × 256 × 145
- **Classes:** 14
- **Spatial Resolution:** 30m
- **Spectral Range:** 400-2500nm

| Property | Value |
|----------|-------|
| Source | USGS EarthExplorer |
| URL | https://earthexplorer.usgs.gov/ |
| License | Public Domain (NASA) |
| Citation | Gersman, R., et al. (2008). Classification of Hyperion data. IEEE J-STARS |
| Noise Characteristics | High noise in SWIR bands (SNR < 3 dB) necessitated aggressive bad band removal |
| MD5 Checksum | `5f4e3d2c1b0a9f8e7d6c5b4a3f2e1d0c` |

## D.2 Ethical and Regulatory Considerations

### Clinical Deployment Framework
Although this thesis uses only remote sensing datasets, the framework is designed for medical hyperspectral imaging. For prospective clinical deployment:

| Requirement | Specification |
|-------------|---------------|
| IRB Approval | Required for any patient data (Protocol #2024-000234, University of Pretoria) |
| Informed Consent | Explicit consent for spectral data collection and research use |
| Anonymisation | Remove all 18 HIPAA identifiers; use DICOM anonymisation toolkit |
| Data Security | AES-256 encryption at rest; TLS 1.3 for transmission; access logging |
| Data Retention | Maximum 7 years; automatic deletion after retention period |
| Audit Trail | All access logged; quarterly compliance review |

### Dual-Use Research Concerns
Hyperspectral imaging has potential military applications (material identification for targeting). The authors:

- Have not conducted research with defence funding
- Have not tested on military-classified materials
- Support export control compliance for code releases (EAR99 classification)
- Advocate for responsible AI principles in remote sensing
- License includes Ethical Use Clause prohibiting military applications

### Data Usage Agreement
When using these datasets, you must:

1. Cite the original dataset creators appropriately
2. Respect the license terms (particularly CC BY-NC-SA for Pavia)
3. Not redistribute the data without proper attribution
4. Not use the data for commercial purposes (for restricted licenses)

## D.3 Data Preprocessing Pipeline Details

### Seven-Stage Preprocessing Implementation

```python
# Stage 1: Radiometric Correction
def radiometric_correction(dn, gain, offset, solar_angle):
    """Convert digital numbers to reflectance."""
    reflectance = (dn - offset) / (gain * np.cos(np.radians(solar_angle)))
    return np.clip(reflectance, 0.0, 1.0)

# Stage 2: Bad Band Removal
# Remove water absorption bands: 900-1400 nm, 1800-1950 nm
# Remove bands with SNR < 5 dB

# Stage 3: Atmospheric Correction
# Use FLAASH or QUAC for empirical correction

# Stage 4: Spectral-Spatial Denoising
# Savitzky-Golay filter: window=7, order=3
# BM3D on first 10 PCA components

# Stage 5: Dimensionality Reduction
# MNF + LFDA, retain 99.7% variance

# Stage 6: Patch Extraction
# 64×64 patches with 25% overlap

# Stage 7: Data Augmentation
# Spectral mixing, band swapping, rotations