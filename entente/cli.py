"""
Command line for invoking utilities in this library.

Example:

    .. code-block:: sh

        python -m entente.cli \\
            examples/vitra/vitra_without_materials_triangulated.obj \\
            examples/vitra/vitra_landmarks.json \\
            examples/vitra/vitra_stretched.obj

"""

import click
from .landmarks.landmark_composite_recipe import DEFAULT_RADIUS


@click.group()
def cli():
    pass


@cli.command()
@click.argument("source_mesh")
@click.argument("landmarks")
@click.argument("target_mesh", nargs=-1, required=True)
@click.option("-o", "--out", help="Output path")
def transfer_landmarks(source_mesh, landmarks, target_mesh, out):
    """
    Transfer landmarks defined in LANDMARKS from SOURCE_MESH to the target
    meshes, which must have vertexwise correspondence.
    """
    import os
    import lacecore
    from .landmarks.landmarker import Landmarker
    from .landmarks.serialization import dump_landmarks

    landmarker = Landmarker.load(source_mesh_path=source_mesh, landmark_path=landmarks)

    for target_mesh_path in target_mesh:
        m = lacecore.load_obj(target_mesh_path, triangulate=True)
        landmarks_on_target_mesh = landmarker.transfer_landmarks_onto(m)
        if out is None:
            filename, _ = os.path.splitext(os.path.basename(target_mesh_path))
            out = filename + ".json"
        dump_landmarks(landmarks_on_target_mesh, out)


@cli.command()
@click.argument("recipe")
@click.argument("output_dir", default="./composite_result")
@click.option("--indicator-radius", default=DEFAULT_RADIUS, show_default=True)
def composite_landmarks(recipe, output_dir, indicator_radius):
    """
    Run the landmark composite recipe in the YAML file RECIPE and write it
    to OUTPUT_DIR.
    """
    import os
    import yaml
    from .landmarks.landmark_composite_recipe import LandmarkCompositeRecipe
    from .landmarks.serialization import dump_landmarks

    recipe_obj = LandmarkCompositeRecipe.load(recipe)

    out_landmarks = os.path.join(output_dir, "landmarks")
    os.makedirs(out_landmarks, exist_ok=True)
    dump_landmarks(recipe_obj.composite_landmarks, f"{out_landmarks}.json")
    with open(f"{out_landmarks}.yml", "w") as f:
        yaml.dump(recipe_obj.to_json(), f)

    recipe_obj.write_reprojected_landmarks(
        output_dir=output_dir, radius=indicator_radius
    )


if __name__ == "__main__":  # pragma: no cover

    def set_path():
        """
        Magically add this project to the module path.
        """
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    set_path()
    cli()
