# Builds assets/model/stone.ply, the "stone": the icosphere (10242 vertices, radius 1)
# smoothly distorted into a lumpy blob with strong curvature of both signs: bumps, dents and saddles
# (about 37% of the area saddle-shaped, 18% concave). Each vertex moves along its sphere normal by
# one octave of improved Perlin noise (C2, so no creases or pits), so the surface stays star-shaped
# (closed, embedded); a mild twist about z then makes the lumps less symmetric.
#
#   blender --background --factory-startup --python assets/model/make_stone.py [-- out.ply]
import os
import sys

import bpy

SUBDIVISIONS = 6  # Blender's 1 is the icosahedron: 20 * 4^5 faces, like the built-in MakeIcosphere(5)
NOISE_SCALE = 1.0  # smaller: more, tighter lumps
STRENGTH = 1.2  # radial displacement is STRENGTH * (noise - 0.5); radii end up in about [0.63, 1.43]
TWIST_DEGREES = 35.0


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    out = argv[0] if argv else os.path.join(os.path.dirname(os.path.abspath(__file__)), "stone.ply")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=SUBDIVISIONS, radius=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object

    tex = bpy.data.textures.new("stone", type="CLOUDS")
    tex.noise_basis = "IMPROVED_PERLIN"  # the original Perlin basis leaves sharp pits and creases
    tex.noise_type = "SOFT_NOISE"
    tex.noise_depth = 0  # one octave: smooth
    tex.noise_scale = NOISE_SCALE

    displace = obj.modifiers.new("displace", type="DISPLACE")
    displace.texture = tex
    displace.texture_coords = "LOCAL"
    displace.direction = "NORMAL"
    displace.mid_level = 0.5
    displace.strength = STRENGTH

    twist = obj.modifiers.new("twist", type="SIMPLE_DEFORM")
    twist.deform_method = "TWIST"
    twist.deform_axis = "Z"
    twist.angle = TWIST_DEGREES * 3.141592653589793 / 180.0

    bpy.ops.object.modifier_apply(modifier=displace.name)
    bpy.ops.object.modifier_apply(modifier=twist.name)
    bpy.ops.object.shade_smooth()

    bpy.ops.wm.ply_export(
        filepath=out,
        export_selected_objects=True,
        apply_modifiers=True,
        export_normals=True,
        export_uv=False,
        export_colors="NONE",
        export_triangulated_mesh=True,
        ascii_format=False,
        forward_axis="Y",
        up_axis="Z",
    )
    print(f"wrote {out}: {len(obj.data.vertices)} vertices, {len(obj.data.polygons)} faces")


main()
