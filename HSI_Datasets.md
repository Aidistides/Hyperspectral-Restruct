# Top Public Hyperspectral Soil & Agriculture Datasets (2024–2025)

Curated high-quality public HSI datasets prioritized for **soil health, contaminants, UAV platforms, and microplastics** detection.  
Perfect for training / fine-tuning the Enotrium 3D CNN pipeline.

| Dataset | Type | Bands / Range | Spatial Resolution | Key Relevance | Direct Link |
|---------|------|---------------|--------------------|---------------|-------------|
| **HYPERVIEW2 (AI4EO Challenge)** | Airborne HSI (Poland) | ~150 bands, VNIR-SWIR | Patch-based | Soil parameters (K, Mg, P₂O₅, pH) — direct match for regenerative soil health | [EOTDL Download](https://www.eotdl.com/datasets/HYPERVIEW2) |
| **LUCAS 2015 Topsoil Spectral Library** (EU) | Lab VNIR-SWIR (point + images) | 400–2500 nm (0.5 nm res) | Point spectra | Pan-European soil properties + heavy metals; widely used benchmark | [ESDAC (free after registration)](https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data) |
| **USGS Spectral Library v7 (splib07)** | Lab reflectance spectra (point) | 0.2–3.0 µm (VNIR-SWIR-MIR) | Point spectra | Authoritative soil, mineral & organic compound reference library; useful for spectral pre-training, wavelength validation, and contaminant signature matching | [USGS ScienceBase](https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d) |
| **Indian Pines** | AVIRIS airborne | 200 bands (400–2500 nm) | 145×145 | Classic agricultural / soil benchmark | [EHUB Hyperspectral Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| **Salinas** | AVIRIS airborne | 224 bands (400–2500 nm) | 512×217 | Agricultural fields + bare soil | [EHUB Hyperspectral Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| **Pavia University** | ROSIS airborne | 103 bands | 610×340 | Urban/soil mix benchmark | [EHUB Hyperspectral Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |
| **Hyperspectral Drone Imagery: Bare Soil Fields (2024)** | UAV (6 NIR bands) | 705–945 nm | Geotiff | Bare soil fields — ideal UAV test set | [Zenodo](https://zenodo.org/records/15297383) |
| **UAV-Borne HSI of Pearl Millet (ICRISAT, 2025)** | UAV VNIR | 282 bands, 400–1000 nm | Canopy level | Crop/soil stress dataset | [OAR@ICRISAT](https://oar.icrisat.org/13212/) |
| **Agro-HSR (2025)** | RGB → HSI reconstruction (sweet potato) | Reconstructed SWIR | 1322 image pairs | Largest agriculture-focused HSI reconstruction dataset | [ScienceDirect / Paper](https://www.sciencedirect.com/science/article/pii/S0168169925012098) |
| **OHID-1 (2025)** | Satellite HSI | Varies (10 full scenes) | Scene level | New large-scale general-purpose HSI benchmark | [Figshare (Nature Scientific Data)](https://figshare.com/articles/online_resource/OHID-1/27966024/8) |
| **HyperPRI** | Underground root HSI + RGB | VNIR | Root-scale | Soil-root interaction (phytoremediation relevant) | [UF ML Lab / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168169924006987) |
| **Plastic Debris Hyperspectral Database (2024)** | Lab SWIR reflectance | 400–2500 nm | Spectra + images | Microplastics in water/soil matrices (PE, PP, PET, etc.) | [Nature Scientific Data](https://www.nature.com/articles/s41597-024-03974-x) |
| **Munsell Soil Color Chart HSI Dataset** | SPECIM IQ lab/field | 204 bands (397–1003 nm) | 20×20 chips + full scenes | Soil color calibration, classification benchmarks, algorithm testing | [Zenodo DOI: 10.5281/zenodo.8143355](https://doi.org/10.5281/zenodo.8143355) |
| **Database of Hyperspectral Images of Phosphorus in Soil** | Bayspec OCIF push-broom | 145 bands (420–1000 nm) | Lab samples (152) | Soil chemistry & nutrient quantification (phosphorus) | [Mendeley Data Version 3](https://data.mendeley.com/datasets/) |
| **Indian Pines AVIRIS (Site 3)** | AVIRIS airborne | 220 bands (400–2500 nm) | ~20m resolution, 2×2 mile | Field-scale agriculture/soil mapping, residue studies | [Purdue University Research Repository](https://purr.purdue.edu/) |

### Enotrium Pipeline
1. **Start here** → HYPERVIEW2 + LUCAS 2015 (soil parameters + ground truth).  
2. Add **Munsell Soil Color Chart** for color calibration and **Phosphorus Database** for nutrient quantification.  
3. Include **Plastic Debris Database** for microplastics pre-training.  
4. Use **Indian Pines AVIRIS (Site 3)** for large-scale field validation.  
5. Fine-tune on your own UAV 400–2500 nm cubes.

### Bonus: SPECCHIO Spectral Libraries
Additional soil spectral collections available as ENVI .sli/.hdr exports from recent Zenodo/Mendeley uploads. Search for "soil hyperspectral ENVI" for the latest spectral library additions.

All datasets are open-access (some require free registration).  
Last updated: April 2026.

---

**Want to add this to README.md?**  
Just drop this line anywhere in your README:

```markdown
## Public Datasets
See [HSI_Datasets.md](HSI_Datasets.md) for the complete curated list with direct download links.
