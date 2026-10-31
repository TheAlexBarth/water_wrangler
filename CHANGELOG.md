# Version 1.0.0
Major Overhaul
- Redesigned format to a more expansive query, format, viz toolset. Still focused on wdft/twdb-coastal products but now more than just baycast. Generalized for environemtnal, hydrology, and hydrodynamic daatsets.
- Modular OOP considerations in design
- Now supports:
    - query of hydrology (twdb and nwm)
    - query of sonde (twdb)
    - query of shapefiles (twdb)
- NOT BACKWARDS COMPATIBLE!

# Version 0.2.1
Minor Changes:
- Added downsampling (in earlier commit) and improved viz tools
Author: Alex Barth

# Version 0.2.0
Major Changes:
- replaced _assign_tri_attrs() with _prep_mesh() and related features. NOT backwards compatible with 0.1.0.
Author Alex Barth

# Version 0.1.0
- Initial build with core features
Author: Alex Barth