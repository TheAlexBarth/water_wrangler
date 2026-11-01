# shared geometry helpers


def _as_geom(boundary):
    """
    Dissolve a GeoDataFrame/GeoSeries to a single geometry; pass geometries through.

    Invalid boundary polygons (some WDFT shapes self-intersect) are repaired with a
    zero-width buffer first, so they do not trip GEOS during the union or the
    within/intersects reach selection.
    """
    geom = getattr(boundary, "geometry", boundary)
    if hasattr(geom, "union_all") or hasattr(geom, "unary_union"):
        if not geom.is_valid.all():
            geom = geom.buffer(0)
        return geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom
