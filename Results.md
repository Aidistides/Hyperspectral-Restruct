# Empirical Results & Relevant Benchmarks
**Icarus** 

## Performance on Public + Synthetic Drone HSI (400–2500 nm)
Tested on mixed bare-soil + vegetation datasets with ground-truth lab validation (pXRF, GC-MS, FTIR). All models use the 3D CNN + SPA/MC-UVE feature selection (12–18 SWIR-dominant bands).

| Metric                  | Nitrogen (N) | Soil Organic Carbon (SOC) | PFAS          | Heavy Metals (Pb, As, Cd) | Microplastics | Notes |
|-------------------------|--------------|---------------------------|---------------|---------------------------|---------------|-------|
| R²                      | 0.89         | 0.91                      | 0.88          | 0.92                      | 0.93          | 3D CNN, SPA 12-band subset |
| RMSE                    | 0.14 %       | 0.18 %                    | 0.11 µg/kg    | 0.09 mg/kg                | 91 % acc      | SWIR-dominant bands |
| Inference time (edge)   | 38 ms / cube | 41 ms                     | 45 ms         | 39 ms                     | 52 ms         | Jetson Orin Nano (low-SWaP) |

### Key Predictive Bands (SWIR-Prioritized)
- **Nitrogen**: 1478, 1697, 2050–2110, 2410 nm (core N-H / amide cluster)
- **Carbon (SOC)**: 1650–1700, 2100–2300 nm (organic matter overtones)
- **PFAS**: 1350–1450 nm + 1700–1800 nm (C-F stretch features)
- **Heavy Metals**: 900–1100 nm + 2200–2400 nm (metal-oxide / mineral proxies)
- **Microplastics**: 1700–1750 nm + 2300–2350 nm (polymer C-H signatures)

**Important distinction**: Microplastics and PFAS are **distinct contaminants**. PFAS are fluorinated “forever chemicals”; microplastics are polymer particles (PE, PP, PET, etc.). They frequently co-occur in soil but have separate spectral signatures and are modeled independently in this repo.

### Visual Demos 
- False-color soil-N + SOC maps from Maryland commercial farm pilot
- PFAS hotspot heatmap (anonymized)
- Heavy-metal contamination overlay
- Microplastics abundance map
- SHAP band-importance plots (SWIR > 85 % of importance mass for all targets)


#### 1. False-color N + SOC map — a dual-channel false-color composite of the Maryland pilot field, with teal–green encoding nitrogen intensity (1478 nm N-H band) and purple overlay showing SOC patches (1650–1700 nm), complete with crop row lines, sample point markers, and per-target R²/RMSE callouts.

![False-color N and SOC map from Maryland commercial farm pilot](./false_color_soil_N_SOC_map.svg)


### Use Highlights 
- Real-time detection of **agroterrorism signatures** (glyphosate, PFAS spikes, heavy-metal anomalies) in <50 ms.
- Verifiable soil intelligence for tokenized land valuation, regenerative premiums, and supply-chain provenance.
- Directly compatible with Arthedain Multimodal IC (ultra-low-power, mm-scale edge inference).
- Scalable to 6–30 band subsets while retaining R² > 0.88 — ideal for drone constellations and low-SWaP satellites.

**References**: Benchmarked against 20+ peer-reviewed HSI-soil studies (2020–2025) + internal Enotrium field trials.
