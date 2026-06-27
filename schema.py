"""MGBIE schema: entity & relation type definitions."""

ENTITY_TYPES = [
    ("CROP", "Crop",            "Grain/cereal crop species or category (e.g. foxtail millet, sorghum, barley)."),
    ("VAR",  "Variety/Cultivar","Named cultivar / line / accession / germplasm (e.g. 晋谷21, B.Tx623, LER 1)."),
    ("TRT",  "Trait",           "Phenotype / agronomic / quality / resistance trait (e.g. plant height, yield, flowering time, salt tolerance)."),
    ("GST",  "Growth Stage",    "Developmental stage / phenology / time point (e.g. flowering stage, seedling stage, 21 DPA, 2018)."),
    ("GENE", "Gene",             "Gene or candidate gene name (e.g. Waxy, SAPK6, SbPUB)."),
    ("QTL",  "QTL",             "Quantitative trait locus name/region (e.g. qPH3.1, QTL for grain yield, stay-green QTLs)."),
    ("MRK",  "Molecular Marker","SSR/SNP/AFLP/Indel/KASP marker (e.g. SSR-Xgwm20, SNPs, AFLP markers, CAPS marker)."),
    ("CHR",  "Chromosome",      "Chromosome / linkage group / genetic interval identifier (e.g. Chr1, 5H, linkage group Mrg21)."),
    ("BM",   "Breeding Method", "Breeding / selection / experimental methodology (e.g. marker-assisted selection, GWAS, RNA-seq)."),
    ("CROSS","Parent/Cross",    "Parent material or cross / mapping population (e.g. A x B, F2 population, RIL, Summer × Kanlow)."),
    ("ABS",  "Abiotic Stress",  "Non-biological stress or treatment (e.g. drought, salt stress, 30% PEG, Na2CO3, water-logging)."),
    ("BIS",  "Biotic Stress",   "Biological stress (pathogen / pest / disease) (e.g. aphid infestation, Striga, BYD, Crown rust)."),
]

ENT_LABELS = [t[0] for t in ENTITY_TYPES]

RELATION_TYPES = [
    ("CON", "CONTAINS",   "(CROP, VAR)",                          "Variety belongs to a crop (crop contains variety)."),
    ("USE", "USES",       "(VAR, BM)",                            "Breeding method used to produce a variety."),
    ("HAS", "HAS",        "(VAR, TRT)",                           "Variety has / exhibits / is evaluated for a trait."),
    ("AFF", "AFFECTS",    "(ABS/GENE/MRK/QTL, TRT)",              "Abiotic stress, gene, marker or QTL affects a trait."),
    ("OCI", "OCCURS_IN",  "(TRT/ABS/BIS, GST)",                   "Trait/stress measured or occurs at a growth stage."),
    ("LOI", "LOCATED_IN", "(MRK/QTL/GENE, CHR)",                  "Marker / QTL / gene is located on a chromosome or region."),
]

REL_LABELS = [t[0] for t in RELATION_TYPES]


def entity_guide() -> str:
    return "\n".join(f"- {code} ({name}): {desc}" for code, name, desc in ENTITY_TYPES)


def relation_guide() -> str:
    return "\n".join(f"- {code} ({name}, typical types {types}): {desc}" for code, name, types, desc in RELATION_TYPES)
