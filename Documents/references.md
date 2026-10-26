# Kidney Ultrasound DL Project: References

Last updated: 2026-10-24

The numbering matches the team literature review (`kidney_ultrasound_DL_review.docx`), so in-text citations [1]–[23] stay valid.

**Baseline:** [24] URI-CADS. Our dataset A is the URI-CADS dataset, so its protocol (5-fold CV, AUC and SP@95 per category) and its results are what we compare against and must beat.

**Complementary references:** everything else supports the work around the baseline.

- **[1]–[8]** Reviews and context papers: clinical motivation and open problems (review Table 1b).
- **[9]–[23]** Primary studies: methods, preprocessing and validation practice (review Table 1). [18] is our dataset B.
- **[S1]–[S13]** Pretrained models we will test, and the prior work URI-CADS compares against.

Abbreviations: CLS classification · GRD grading · SEG segmentation · FUN function prediction · QA quality selection · DATA dataset release · HO hold-out test · CV cross-validation · TL transfer learning (ImageNet) · n.r. not reported.

---

# Baseline

## [24] URI-CADS: A Fully Automated Computer-Aided Diagnosis System for Ultrasound Renal Imaging

- **Citation:** Molina-Moreno M., González-Díaz I., Rivera Gorrín M., Burguera Vion V., Díaz-de-María F. *Journal of Imaging Informatics in Medicine*, 2024. https://doi.org/10.1007/s10278-024-01055-4
- **Dates:** received 26 Dec 2023, accepted 14 Feb 2024, published online 27 Feb 2024.
- **Affiliations:** Dept. of Signal Theory and Communications, Universidad Carlos III de Madrid; Hospital Ramón y Cajal and IRYCIS, Madrid; Universidad de Alcalá.
- **Code and data:** https://github.com/miguel55/URI-CADS
- **Licence:** CC BY 4.0 (open access).
- **Ethics:** approved by the IRB of Hospital Universitario Ramón y Cajal; informed consent waived (retrospective study).

#### Data (same as our dataset A)

| Item | Value |
|---|---|
| Images | 1,985 B-mode renal images, JPG, sizes 375–600 × 382–810 px |
| Classes | 450 healthy, 1,535 pathological |
| Collection | Retrospective, 2009–2018, one image per clinical case, patients over 18, sex-balanced |
| Scanner | Toshiba Xario-660a, 3 MHz and 3.5 MHz convex probes |
| Views | 93% transverse, 7% longitudinal (our `View: 0` = transverse) |
| Annotators | Two nephrologists (25 and 10 years of experience), independent annotation, consensus on disagreement |
| Annotations | Kidney polygon mask, global pathologies, lesion bounding boxes with class |

#### Label key (our label files → paper taxonomy)

| Our label | Pathology | Count | Grouped category |
|---|---|---|---|
| Box class 1 | Lithiasis (LIT) | 110 boxes | Others (O) |
| Box class 2 | Simple cyst (SCY) | 1,073 boxes | Cyst (C) |
| Box class 3 | Complicated cyst (CCY) | 92 boxes | Cyst (C) |
| Box class 4 | Pyramid (PYR) | 384 boxes | Pyramid (PYR) |
| Box class 5 | Angiomyolipoma (ANG) | 34 boxes | Others (O) |
| Box class 6 | Solid renal mass (SRM) | 11 boxes | Others (O) |
| Box class 7 | Hydronephrosis (HYD) | 618 boxes | Hydronephrosis (HYD) |
| Box class 8 | Cortex thinning (CT) | 111 boxes | Others (O) |
| Box class 9 | Cortex eschar (CE) | 45 boxes | Others (O) |
| Global flag 1 | Poor corticomedullary differentiation (PCD) | 595 images | PCD (global) |
| Global flag 2 | Hyper-echoic cortex (HC) | 509 images | HC (global) |

The final taxonomy has 6 categories (HC, PCD, C, PYR, HYD, O) plus healthy (H).

#### Method

- **Backbone:** ResNet-50, ImageNet-pretrained, with two Feature Pyramid Networks.
- **Three branches:** kidney segmentation (Mask R-CNN), image-level classification (one binary cross-entropy per category), and region-level lesion detection (Faster R-CNN).
- **Diagnosis Generation Module:** combines the image and region scores as p = α · p_image + (1 − α) · p_region. α is learned per category (URI-CADS-C) or per case by attention (URI-CADS-Att, the final model).
- **Region aggregation:** max over the detected regions of each category (best of max / mean / LME / area).
- **Loss:** L = L_seg + L_clas + L_reg + L_dgm, trained end to end.
- **Augmentation:** rotation, gamma, translation, zoom (supervised by the nephrologists).
- **Validation:** 5-fold cross-validation (3 train, 1 validation, 1 test per repetition).
- **Metrics:** AUC (sensitivity–specificity ROC) and specificity at 95% sensitivity (SP@95) per category; IoU and Dice for segmentation.

#### Results to beat

Classification, URI-CADS-Att (Table 6):

| Category | AUC (%) | SP@95 (%) |
|---|---|---|
| **Healthy vs pathological** | **87.41** | **60.59** |
| HC (global) | 78.65 | 29.28 |
| PCD (global) | 84.15 | 43.63 |
| Cyst | 79.59 | 28.19 |
| Pyramid | 86.61 | 48.19 |
| Hydronephrosis | 93.04 | 63.19 |
| Others | 69.32 | 14.05 |
| **Multi-class average** | **81.90** | **37.76** |

Kidney segmentation on 1,985 images (Table 5):

| Method | IoU (%) | Dice (%) |
|---|---|---|
| DeepLabv3+ | 81.80 | 89.34 |
| TN-SCUI2020 winner | 79.31 | 87.23 |
| URI-CADS-Segmentation (segmentation only) | **84.99** | **91.23** |
| URI-CADS (full multi-task) | 81.41 | 89.38 |

Ablation (Table 3, AUC %): image branch only 76.83 average / 85.34 binary; region branch only 72.52 / 77.53; category fusion 79.62 / 87.21; attention fusion 81.90 / 87.41.

Compared methods on this dataset (Table 6, binary AUC %): Sudharson-orig 47.14, Sudharson-improved 64.37, TN-SCUI2020 77.39.

#### Limitations (our reading)

- ImageNet-pretrained backbone; no ultrasound-specific pretraining.
- Lesion boxes only, no pixel masks ("we lack region masks").
- "Others" category weak (69.32% AUC).
- Single centre and single scanner; no external validation.
- No confidence intervals or calibration reported.
- Near-duplicates and on-screen text not discussed.
- Kidney size and shape listed as future work.

#### Our findings on this dataset

- 180 images in 87 near-duplicate groups (50 identical pairs, 43 consecutive-frame pairs); see `duplicate_groups.csv`.
- 31 of the 87 groups have inconsistent labels between copies; #4 (pathological) and #296 (healthy) are the same image.
- Some images carry diagnosis text on screen (e.g. "QUISTE", "HIDRONEFROSIS", "DESESTRUCTURADO").

---

# Complementary references

## [1]–[8] Reviews and context papers

| # | Year | Reference | Type | Key takeaway for us |
|---|---|---|---|---|
| 1 | 2025 | Deep learning in renal ultrasound: applications, challenges, and future outlook. *Front Oncol* 2025. doi:10.3389/fonc.2025.1730628 | Systematic review (98 articles) | Open problems: data quality, interpretability, generalisation, clinical integration |
| 2 | 2024 | A narrative review on the application of artificial intelligence in renal ultrasound. *Front Oncol* 2024. doi:10.3389/fonc.2023.1252630 | Narrative review | AI can reduce the large interobserver variability in renal ultrasound |
| 3 | 2023 | Artificial intelligence-aided ultrasound in renal diseases: a systematic review. *Quant Imaging Med Surg*. PMC10240007 | Systematic review | CKD and quantitative hydronephrosis are the most promising AI applications |
| 4 | 2026 | Koratala A, Ferreira F, Diniz H. Letter to the Editor: Artificial intelligence in nephrology point-of-care ultrasonography – opportunities, limitations, and future directions. *World J Nephrol* 2026;15(3):124607 | Letter | Correct image recognition is not correct interpretation; AI should support clinicians |
| 5 | 2025 | Point-of-care kidney and genitourinary ultrasound in adults: image acquisition. *J Vis Exp (JoVE)*, article 66802 | Methods article | Kidney imaging is unfamiliar to most clinicians, including in primary care |
| 6 | 2018 | Nixon G, Blattner K, Muirhead J, Kerse N. Rural point-of-care ultrasound of the kidney and bladder: quality and effect on patient management. *J Prim Health Care* 2018;10(4):324–330 | Clinical study | Kidney POCUS is already used in primary care |
| 7 | 2022 | Sonko ML, Arnold TC, Kuznetsov IA. Machine learning in point of care ultrasound. *POCUS J* 2022;7(Kidney) | Review | Scanners increasingly run on-device ML for bedside results |
| 8 | 2026 | Satyagraha P, Budaya TN, Viosandy J. Advancements in artificial intelligence for detecting urinary tract stones: a systematic review of imaging-based approaches. doi:10.1177/20514158261454815 | Systematic review | Ultrasound AI is a practical, low-cost option for resource-limited settings |

---

## [9]–[23] Primary studies

| # | Year | First author | Task | Pathology / classes | Population | Images (patients) | Best model | Best result | Validation |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 2019 | Kuo | FUN, CLS | eGFR; CKD (eGFR < 60) | Adult | 4,505 (n.r.) | ResNet (TL, bagging) | Acc 85.6% (CKD) | HO |
| 10 | 2020 | Sudharson | CLS | Normal, cyst, stone, tumour | n.r. | n.r. | Ensemble (ResNet-101, ShuffleNet, MobileNetV2 + SVM) | Acc 96.5% (quality images); 95.6% (noisy) | HO |
| 11 | 2020 | Hao | CLS | CKD screening | n.r. | 226 (n.r.) | CNN + texture branch (TL) | Acc 96.0%; Sens 99.4% | n.r. |
| 12 | 2020 | Athavale | GRD | IFTA grades 1–4 | Adult | 6,135 (352) | VGG-19 features + XGBoost; U-Net (IoU 0.91) | Acc 86.8%; F1 83.9% | HO |
| 13 | 2020 | Chen | GRD | CKD stages (eGFR) | Adult | n.r. | Texture + SVM | n.r. | n.r. |
| 14 | 2021 | Kim | GRD | Normal, mild/moderate CKD, severe CKD | n.r. | 741 (n.r.) | GLCM + ANN | Acc 95.4% | n.r. |
| 15 | 2022 | Tsai | CLS | Normal vs abnormal (stone, cyst, hyperechogenicity, SOL, hydronephrosis) | Pediatric | 1,599 (n.r.) | ResNet-50 (TL) | Acc 92.9%; AUC 0.959 | HO |
| 16 | 2022 | Priyanka | CLS | Stone, cyst, hydronephrosis (likely) | n.r. | n.r. | AlexNet (TL) | CNN > GAONN (no value) | n.r. |
| 17 | 2022 | HARP study | SEG, GRD | Hydronephrosis area / parenchyma ratio | Pediatric | 168 (195) | Ensemble of 5 nets (DeepLabV3+ / UNet++) | Dice 0.911 | 4-fold CV |
| 18 | 2022 | Singla | DATA, SEG | Kidney anatomy (capsule, cortex, medulla, CEC) | Adult | 514 (514), 8 scanners | nnU-Net | Dice 0.85 capsule; 0.69 mean | Public set |
| 19 | 2023 | Lien | SEG, CLS | Moderate/severe hydronephrosis vs normal | Adult | 5,090 (362) | Res-UNet | Acc 94.6% | n.r. |
| 20 | 2024 | EUO HN study | CLS, SEG | Hydronephrosis vs none | Adult | n.r.; 200 test, 3 devices | DeepLabV3-ResNet50/101; AlexNet, ResNet, GoogLeNet | n.r. | HO (200 new images) |
| 21 | 2025 | Seraj | QA | Frame quality (good, bad, null) | n.r. | 1,203 (211) | YOLO11x | Acc 90% ± 5.9 | 5-fold CV |
| 22 | 2026 | Alkhaldi | SEG, CLS | Normal, CKD, AKI | n.r. | 293 (293) + 9,337 unlabelled | ResNet50 dual-input; semi-supervised U-Net (Dice 0.91) | Acc 82.9%; AUC 0.913 | HO (3 seeds) |
| 23 | 2026 | Kanchanamala | CLS | Stone | n.r. | n.r. | KSOMFHNet (hybrid), DRRN denoising | Acc 92.0%; Sens 90.5%; Spec 92.2% | n.r. |

### Full citations

9. Kuo CC, et al. Automation of the kidney function prediction and classification through ultrasound-based kidney imaging using deep learning. *npj Digit Med* 2019.
10. Sudharson S, Kokil P. An ensemble of deep neural networks for kidney ultrasound image classification. *Comput Methods Programs Biomed* 2020;197:105709.
11. Hao PY, Xu ZY, Tian SY, et al. Texture branch network for chronic kidney disease screening based on ultrasound images. *Front Inf Technol Electron Eng* 2020;21(8):1161–1170.
12. Athavale AM, et al. Deep learning to predict interstitial fibrosis and tubular atrophy from kidney ultrasound images. *medRxiv* 2020. doi:10.1101/2020.08.17.20176958.
13. Chen CJ, Pai TW, Hsu HH, et al. Prediction of chronic kidney disease stages by renal ultrasound imaging. *Enterp Inf Syst* 2020;14(2):178–195.
14. Kim DH, Ye SY. Classification of chronic kidney disease in sonography using the GLCM and artificial neural network. *Diagnostics* 2021;11(5):864.
15. Tsai MC, Lu HHS, Chang YC, Huang YC, Fu LS. Automatic screening of pediatric renal ultrasound abnormalities: deep learning and transfer learning approach. *JMIR Med Inform* 2022;10(11):e40878.
16. Priyanka, Kumar D. Kidney image classification using transfer learning with convolutional neural network. *Int J Comput Vis Robot* 2022;12(6).
17. Deep-learning segmentation of ultrasound images for automated calculation of the hydronephrosis area to renal parenchyma ratio. *Investig Clin Urol* 2022.
18. Singla R, et al. The Open Kidney Ultrasound Data Set. *arXiv* 2022;2206.06657. https://arxiv.org/abs/2206.06657
19. Lien WC, et al. Detecting hydronephrosis through ultrasound images using state-of-the-art deep learning models. *Ultrasound Med Biol* 2023;49(3).
20. Harnessing artificial intelligence for enhanced renal analysis: automated detection of hydronephrosis and precise kidney segmentation. *Eur Urol Open Sci* 2024.
21. Seraj A, et al. Deep learning framework for automated frame selection in kidney ultrasound. *Sci Rep* 2025;15:41940.
22. Alkhaldi N, et al. Automating kidney disease diagnosis: classifying acute and chronic kidney disease from ultrasound images. *Eng Technol Appl Sci Res* 2026;16(1).
23. Kanchanamala P, et al. Ultrasound-based kidney stone classification using Kronecker self-organizing map forward harmonic network. *Ultrason Imaging* 2026.

### Notes on [18], our dataset B

- The review records 514 images from 514 patients and 8 scanners. The copy we received (`kidneyUS_images_25_june_2025`) has 534 images from 511 patients, so it is a later release.
- Licence: non-commercial, registration required (per the review).
- Published results: capsule Dice 0.85; cortex 0.52, medulla 0.59. Inter-annotator Dice in our copy (our calculation): capsule 0.94, central echo complex 0.86, medulla 0.67, cortex 0.57.
- Our use: external test for kidney segmentation only (no pathology labels).

### Main gaps across [9]–[23] (from the review)

- Only 5 of 14 report patient counts; 1 of 14 reports devices.
- Mostly single hold-out splits; 2 use k-fold CV; few split by patient.
- No study validates on an external hospital or on primary-care images.
- Only [18] releases data and code.

---

## [S1]–[S13] Pretrained models and prior work

### Pretrained models

- **[S1] UltraSam.** Meyer A., Murali A., Mutter D., Padoy N. *UltraSam: A Foundation Model for Ultrasound using Large Open-Access Segmentation Datasets.* arXiv:2411.16222. https://arxiv.org/abs/2411.16222. Code: https://github.com/CAMMA-public/UltraSam
  - SAM trained on US-43d (43 open ultrasound datasets); point and box prompts; its encoder beat ImageNet, SAM and MedSAM initialisations on downstream tasks.
- **[S2] USFM.** Jiao J. et al. *USFM: A Universal Ultrasound Foundation Model Generalized to Tasks and Organs towards Label Efficient Image Analysis.* Medical Image Analysis. arXiv:2401.00153. https://arxiv.org/abs/2401.00153
  - Self-supervised on more than 2 million multi-organ, multi-centre, multi-device ultrasound images; robust with 20% of annotations.
- **[S3] MedSAM.** Ma J. et al. *Segment Anything in Medical Images.* Nature Communications, 2024. https://doi.org/10.1038/s41467-024-44824-z. Code: https://github.com/bowang-lab/MedSAM. Weights: https://zenodo.org/records/10689643
- **[S4] MedSAM2.** Bo Wang Lab. *MedSAM2: Segment Anything in 3D Medical Images and Videos.* https://github.com/bowang-lab/MedSAM2
- **[S5] UltraFedFM.** Jiang Y. et al. *From pretraining to privacy: federated ultrasound foundation model with self-supervised learning.* npj Digital Medicine 8:714, 2025. Weights (mirror, Apache-2.0): https://huggingface.co/fmassimo/UltraFedFM
- **[S6] SonoBase.** Qin C. et al. *Open ultrasound foundation model for robust segmentation and clinical measurement across heterogeneous settings.* arXiv:2609.19230. Code: https://github.com/AlfredQin/sonobase. Weights (CC BY-NC 4.0): https://huggingface.co/AlfredQin/sonobase
- **[S7] OpenUS.** Zheng X. et al. *OpenUS: A Fully Open-Source Foundation Model for Ultrasound Image Analysis via Self-Adaptive Masked Contrastive Learning.* arXiv:2511.11510. https://github.com/XZheng0427/OpenUS
- **[S8] OmniRad-small.** Radiology foundation model (DINOv2-style, CT/MRI/X-ray/ultrasound), CC BY 4.0. https://huggingface.co/Snarcy/OmniRad-small

### Prior work cited in URI-CADS [24]

- **[S9] Ravishankar H. et al.** Learning and incorporating shape models for semantic segmentation. MICCAI 2017, pp. 203–211. Kidney segmentation, 171 test images, Dice 84%.
- **[S10] Yin S. et al.** Automatic kidney segmentation in ultrasound images using subsequent boundary distance regression and pixelwise classification networks. Medical Image Analysis 60:101602, 2020. 289 images, Dice 93.03%.
- **[S11] Chen G. et al.** SDFNet: Automatic segmentation of kidney ultrasound images using multi-scale low-level structural feature. Expert Systems with Applications 185:115619, 2021. 50 test images, Jaccard 91%.
- **[S12] Smail L. C. et al.** Using deep learning algorithms to grade hydronephrosis severity: toward a clinical adjunct. Frontiers in Pediatrics 8, 2020. 2,420 images, 51% accuracy.
- **[S13] Wang M.** 1st place solution for the Thyroid Nodule Segmentation and Classification in Ultrasound Images Grand Challenge (TN-SCUI 2020). https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st

Sudharson and Kokil, also cited by URI-CADS, is [10] above. On our dataset, URI-CADS reports binary AUC 47.14% for the original method and 64.37% for an improved version.
