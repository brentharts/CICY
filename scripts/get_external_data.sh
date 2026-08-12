#!/usr/bin/env bash
# scripts/get_external_data.sh
#
# Downloads the external datasets used (or usable) by
# pyCICY/theories/parity.py and the crossover supplementary paper.
# Run from the repository root:  bash scripts/get_external_data.sh
#
# Section 1 is required for the Planck plik-lite layer-two search and is
# fetchable from GitHub (the analysis environment can reach it too; a
# vendored copy may already exist under external/). Sections 2-4 are hosted
# outside GitHub/PyPI and must be fetched from a normal network, then
# committed so the analysis environment can pull them.

set -e
mkdir -p external
cd external

# ---------------------------------------------------------------------
# 1. Planck 2018 plik-lite band powers (TT 215 bins ell 30-2508, TE/EE
#    199 bins ell 30-1996, covariance, bin weights). ~7 MB.
#    Source: github.com/heatherprince/planck-lite-py (data/ subdirectory)
# ---------------------------------------------------------------------
if [ ! -d planck_lite_data ]; then
    git clone --depth 1 https://github.com/heatherprince/planck-lite-py.git _plp
    cp -r _plp/data planck_lite_data
    cp _plp/planck_lite_py.py .
    rm -rf _plp
    echo "planck_lite_data: done"
else
    echo "planck_lite_data: already present"
fi

# ---------------------------------------------------------------------
# 2. BICEP/Keck 2018 (BK18) baseline likelihood band powers: the B-mode
#    data behind r < 0.036, needed if the tensor-chirality channel opens.
#    ~10 MB. NOT on GitHub; fetch and commit.
# ---------------------------------------------------------------------
if [ ! -d BK18_cosmomc ]; then
    wget -q --show-progress \
        http://bicepkeck.org/BK18_datarelease/BK18_cosmomc.tgz \
        -O BK18_cosmomc.tgz \
        && tar xzf BK18_cosmomc.tgz && rm BK18_cosmomc.tgz \
        && echo "BK18: done" \
        || echo "BK18: FETCH FAILED (get it at bicepkeck.org, tab 'Data')"
fi

# ---------------------------------------------------------------------
# 3. Eskilt 2026 joint ACT DR6 + Planck PR4 birefringence analysis code
#    and EB band powers (the beta = 0.277 +/- 0.057 deg measurement).
#    The paper states its reproduction code is public; the repository is
#    linked from arXiv:2608.06480. Clone it here as eskilt_2026_joint.
#    (GitHub-hosted, so this may also work from the analysis
#    environment once the exact URL is known.)
# ---------------------------------------------------------------------
echo "eskilt_2026_joint: clone the repository linked from"
echo "  https://arxiv.org/abs/2608.06480  into external/eskilt_2026_joint"

# ---------------------------------------------------------------------
# 4. Planck NPIPE (PR4) EB power spectra used by Eskilt-Komatsu 2022
#    (beta = 0.342 +/- 0.09 deg): distributed with arXiv:2205.13962's
#    analysis code. Clone into external/eskilt_komatsu_2022 if a direct
#    EB-band-power reanalysis is wanted.
# ---------------------------------------------------------------------
echo "eskilt_komatsu_2022: see code link in arXiv:2205.13962"

echo
echo "After fetching, commit external/ (minus anything > ~50 MB) and push."
