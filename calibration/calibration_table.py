"""Shared check: arm stays >= 20 mm above the table along a path (the sim does not model objects on the table)."""
def table_ok(scene, path, clearance=0.020):
    from hand_eye import table_clearance
    return table_clearance(scene, path) >= clearance
