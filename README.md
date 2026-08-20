## Overview

This library provides hash-to-curve implementations for BLS curves, built on a modified version of the RELIC toolkit ("RELIC-updated"). The modifications focus on prime field operations required for hash-to-curve constructions and include:

- Two new curves: **BLS24-559** and **BLS48-571**
- Two new parametrizations: **BLS24-509** and **BLS48-575**

### Hash-to-Curve Constructions

We provide implementations of three hash-to-curve constructions:

- **Elligator-SK**
- **FF-SK**
- **KV hash** (proposed by Koshelev)

For the **WB hash** (proposed by Wahby and Boneh), the corresponding isogeny maps are required — we have included all remaining isogeny maps needed for this construction.

### Square Root Computation

Square root computation is one of the primary bottlenecks in hash-to-curve constructions. We implement three Tonelli–Shanks variants:

- **TSB** — proposed by Bernstein
- **TSS** — proposed by Sarkar
- **TSP** — proposed by Pornin

These are implemented and benchmarked for the **BLS12-377**, **BLS24-509-SNARK**, **NIST P-224**, and **STARK** prime fields. Our improved variant of TSB is referred to as **TSB-SK**. In addition, support for the STARK prime field and the corresponding STARK curve is included in RELIC-updated.

## Acknowledgments

This work builds on the original **RELIC Toolkit**, proposed by Diego F. Aranha et al., available at [https://github.com/relic-toolkit/relic](https://github.com/relic-toolkit/relic.git).
