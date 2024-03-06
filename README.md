# Warming levels from CMIP6 and CMIP5 

Code to calculate the degrees of warming for a given project (CMIP5/CMIP6), experiment (historical, RCP8.5, SSP5-8.5), and baseline vs future year ranges. Also code to calculate the year at which a certain warming level is reached, using a running mean global average temperature.

CMIP6 and CMIP5 warming levels are based on the IPCC Repository supporting the implementation of FAIR principles in the IPCC-WG1 Atlas, available at https://github.com/IPCC-WG1/Atlas and licensed by CC-BY-4.0:

> Iturbide, M., Fernández, J., Gutiérrez, J.M., Bedia, J., Cimadevilla, E., Díez-Sierra, J., Manzanas, R., Casanueva, A., Baño-Medina, J., Milovac, J., Herrera, S., Cofiño, A.S., San Martín, D., García-Díez, M., Hauser, M., Huard, D., Yelekci, Ö. (2021) Repository supporting the implementation of FAIR principles in the IPCC-WG1 Atlas. Zenodo, DOI: 10.5281/zenodo.3691645. Available from: https://github.com/IPCC-WG1/Atlas 

The approach is similar to that used in https://github.com/IPCC-WG1/Atlas/tree/main/warming-levels and https://github.com/mathause/cmip_warming_levels.

The main document is the notebook [warming_levels.ipynb](warming_levels.ipynb) which explains the method and shows examples.

This code is released under the GPL-3.0 license with the exception of the data under the data directory which are covered by the CC-BY-4.0 license (see data/README.md).
