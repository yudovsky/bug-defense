# References

Source papers for the benchmark suite in this directory. Citations were checked against
CrossRef / PubMed and the full text.

## Files in this folder

| File | Citation | Licence | Used for |
|---|---|---|---|
| `Shimojo2023_SciRep_laser_ablation_MC_DOP.pdf` | Shimojo Y, Sudo K, Nishimura T, Ozawa T, Tsuruta D, Awazu K. *Transient simulation of laser ablation based on Monte Carlo light transport with dynamic optical properties model.* **Sci Rep** 13, 11898 (2023). [doi:10.1038/s41598-023-39026-4](https://doi.org/10.1038/s41598-023-39026-4) | CC BY 4.0 | Validation tier (Figs. 5 and 7, digitised) |
| `Shimojo2023_SciRep_Supplementary.docx` | Supplementary Information to the above (Table S1 material properties; measurement definitions) | CC BY 4.0 | All model inputs for the Shimojo replication |
| `McMillan2021_LSM_predictive_MCRT_fractional_CO2.pdf` | McMillan L, O'Mahoney P, Feng K, Zheng K, Barnard IRM, Li C, Ibbotson S, Eadie E, Brown CTA, Wood K. *Development of a predictive Monte Carlo radiative transfer model for ablative fractional skin lasers.* **Lasers Surg Med** 53(5), 731–740 (2021). [doi:10.1002/lsm.23335](https://doi.org/10.1002/lsm.23335) | CC BY (St Andrews repository copy) | Reproduced (Figs. 3 & 7 digitised; engine does **not** match, see report §3) |
| `Kim2023_JCDE_laser_tissue_skin_lesions_fulltext.html` | Kim H-J, Um S-H, Kang YG, Shin M, Jeon H, Kim B-M, Lee D, Yoon K. *Laser–tissue interaction simulation considering skin-specific data to predict photothermal damage lesions during laser irradiation.* **J Comput Des Eng** 10(3), 947–958 (2023). [doi:10.1093/jcde/qwad033](https://doi.org/10.1093/jcde/qwad033) — Internet Archive snapshot of the publisher HTML (PDF is not retrievable) | Gold OA | Reproduced with exposure time fitted (report §4) |
| `Keller2016_SciRep_laser_mortality_Anopheles.pdf` | Keller MD, Leahy DJ, Norton BJ, Johanson E ("3ric" in the PDF), Mullen ER, Marvit M, Makagon A. *Laser induced mortality of Anopheles stephensi mosquitoes.* **Sci Rep** 6, 20936 (2016). [doi:10.1038/srep20936](https://doi.org/10.1038/srep20936) | CC BY 4.0 | Table 1 plotted; thermal-damage physics check (report §5) |

## Cited but not redistributed here

- Wang L, Jacques SL. *Monte Carlo Modeling of Light Transport in Multi-layered Tissues in Standard C.* Univ. Texas M. D. Anderson Cancer Center (1992). Free from OMLC: https://omlc.org/software/mc/mcman.pdf. Software paper: Wang L, Jacques SL, Zheng L. *MCML—Monte Carlo modeling of light transport in multi-layered tissues.* **Comput Methods Programs Biomed** 47(2), 131–146 (1995). [doi:10.1016/0169-2607(95)01640-F](https://doi.org/10.1016/0169-2607(95)01640-F) — Tables 6.1/6.2 used for V1/V2.

- Sagi-Dolev AM, Avidor-Zehavi A, Shitzer A, Gerstmann M, Akselrod S, Katzir A. *Heating of biological tissue by laser irradiation: temperature distribution during laser ablation.* **Opt Eng** 31(7), 1425–1431 (1992). [doi:10.1117/12.58733](https://doi.org/10.1117/12.58733). Only the abstract could be verified; it is **not** used as a benchmark. The steady-state ablation-velocity test (V4) is a textbook energy balance, not a reproduction of this paper.
- van de Hulst HC. *Multiple Light Scattering*, Vol. II. Academic Press (1980) — slab Rd/Tt values (via MCML Table 6.1).
- Giovanelli RG. *Reflection by semi-infinite diffusers.* **Optica Acta** 2, 153–162 (1955) — semi-infinite Rd, n = 1.5 (via MCML Table 6.2).
- Prahl SA, Keijzer M, Jacques SL, Welch AJ. *A Monte Carlo model of light propagation in tissue.* SPIE IS 5, 102–111 (1989) — comparison values in MCML Tables 6.1/6.2.
- Chandrasekhar S. *Radiative Transfer.* Oxford (1950) / Dover (1960), Ch. V — H-function; exact semi-infinite Rd for V2 (computed in code).
- Carslaw HS, Jaeger JC. *Conduction of Heat in Solids*, 2nd ed. Oxford (1959) — closed-form heating solution used in V3.

## How each paper is used (details and verdicts in `../README.md`)

- **Shimojo et al. 2023** — Figs. 5/7 digitised; scanned-beam replication with Supplementary Table S1 inputs.
- **McMillan et al. 2021** — Figs. 3/7 digitised exactly (vector figure); single-pixel replication with the paper's stated inputs. **Does not reproduce** (engine 2–3× too deep); the dried-tissue absorption is not given in the paper and is varied as a sensitivity.
- **Kim et al. 2023** — per-specimen depths/areas from the text. The irradiation time/pulse count is not reported, so it is the one fitted parameter per group; depth is therefore *calibrated*, top-view area is the prediction.
- **Keller et al. 2016** — Table 1 plotted; mosquito mortality is not reproducible, so only the paper's thermal-confinement interpretation is checked (Arrhenius damage fluence for the three water-absorbed CO₂ rows). No beetle-cuticle threshold exists in it.
- **Sagi-Dolev et al. 1992** — paywalled; only the abstract's five-stage model description is compared (no numbers).
- **MCML manual, van de Hulst, Giovanelli, Prahl** — Tables 6.1/6.2 (rows attributed to each author in the manual).
- **Chandrasekhar** — exact H-function reflectance recomputed (book tables not on hand).
- **Carslaw & Jaeger** — closed-form image-source heating solution.

Any further reference must have its DOI resolved and its source file added here **before** a comparison is made.
