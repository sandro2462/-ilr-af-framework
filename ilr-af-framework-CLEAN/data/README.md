# Data availability

## Patient-level ILR data (controlled access)

The underlying implantable loop recorder (ILR) records are **personal health data** and are
**not distributed in this repository**. They were collected under a data use agreement and are
available from the corresponding author **upon reasonable request, subject to institutional
data-sharing agreements, ethics-committee approval, and applicable privacy regulations
(including the EU General Data Protection Regulation)**.

The raw daily records contain quasi-identifiers (exact calendar dates and exact age in a small
single-centre cohort) and are therefore pseudonymised, not anonymised. They cannot be released
under an open licence without prior de-identification (removal of calendar dates, age banding
and top-coding) and explicit approval from the ethics committee and the data controller.

Requests: sandro.gelsomino@gmail.com

## Synthetic data (`data/synthetic/`)

Synthetic time series generated under the clinician-supervised TIMA protocol are non-identifiable
by construction (they do not correspond to any real individual) and may be used freely for method
development and reproduction of the augmentation experiments.
