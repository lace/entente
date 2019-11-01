"""
Command line for invoking utilities in this library.

Example:

    .. code-block:: sh

        python -m entente.cli examples/vitra/vitra.obj examples/vitra/vitra.pp \\
            examples/vitra/vitra_stretched.obj

"""

import click


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
    from lace.mesh import Mesh
    from lace.serialization import meshlab_pickedpoints
    from .landmarks.landmarker import Landmarker

    landmarker = Landmarker.load(source_mesh_path=source_mesh, landmark_path=landmarks)

    for target_mesh_path in target_mesh:
        m = Mesh(filename=target_mesh_path)
        landmarks_on_target_mesh = landmarker.transfer_landmarks_onto(m)
        if out is None:
            filename, _ = os.path.splitext(os.path.basename(target_mesh_path))
            out = filename + ".pp"
        meshlab_pickedpoints.dump(landmarks_on_target_mesh, out)


@cli.command()
@click.argument("recipe")
@click.argument("output_dir", default="./composite_result")
def composite_landmarks(recipe, output_dir):
    """
    Run the landmark composite recipe in the YAML file RECIPE and write it
    to OUTPUT_DIR.
    """
    import os
    import yaml
    from lace.serialization import meshlab_pickedpoints
    from .landmarks.landmark_composite_recipe import LandmarkCompositeReceipe

    recipe_obj = LandmarkCompositeReceipe.load(recipe)

    composite_landmarks = recipe_obj.composite_landmarks
    out_landmarks = os.path.join(output_dir, "landmarks")
    yaml.dump(composite_landmarks, "{}.yml".format(out_landmarks))
    meshlab_pickedpoints.dump(composite_landmarks, "{}.pp".format(out_landmarks))

    reprojected_landmraks = recipe_obj.reprojected_landmarks


if __name__ == "__main__":  # pragma: no cover
    transfer_landmarks()
