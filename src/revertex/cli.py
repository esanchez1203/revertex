from __future__ import annotations

import argparse
import logging
import random

import pyg4ometry

from revertex import core, utils
from revertex.generators import (
    alpha_n,
    axion,
    beta,
    borehole,
    musun_gs,
    shell,
    surface,
)
from revertex.utils import setup_log

log = logging.getLogger(__name__)


def cli(args=None) -> None:
    parser = argparse.ArgumentParser(
        prog="revertex",
        description="%(prog)s command line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=None,
        type=int,
        help="Seed for rng",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # beta spectra
    beta_parser = subparsers.add_parser(
        "beta-kin", help="Generate beta kinematics from a csv file."
    )

    beta_parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        type=str,
        help="Path to the file with the spectrum to sample",
    )
    beta_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    beta_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )
    beta_parser.add_argument(
        "--eunit",
        "-e",
        required=True,
        type=str,
        help="Unit for the energy",
    )

    hpge_surface_parser = subparsers.add_parser(
        "hpge-surf-pos", help="Generate samples from the surface of the HPGes"
    )

    hpge_surface_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )
    hpge_surface_parser.add_argument(
        "--surface-type",
        "-t",
        required=True,
        type=str,
        help="Type of surface",
    )
    hpge_surface_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_surface_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_surface_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )

    hpge_shell_parser = subparsers.add_parser(
        "hpge-shell-pos", help="Generate samples from the shell of the HPGes"
    )

    hpge_shell_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )
    hpge_shell_parser.add_argument(
        "--surface-type",
        "-t",
        required=True,
        type=str,
        help="Type of surface",
    )
    hpge_shell_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_shell_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_shell_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )
    hpge_shell_parser.add_argument(
        "--radius",
        "-r",
        required=True,
        type=float,
        help="Radius of the shell to generate in",
    )

    hpge_borehole_parser = subparsers.add_parser(
        "hpge-borehole-pos", help="Generate samples from the borehole of the HPGes"
    )

    hpge_borehole_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )

    hpge_borehole_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_borehole_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_borehole_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )

    alpha_n_parser = subparsers.add_parser(
        "alpha-n-kin",
        help="Generate neutron and gamma kinematics from SaG4n.",
        description="""
        There are three ways to use this option:
        (1) providing a valid SaG4n input file in `input-file-sag4n` (requires `output-file`).
        (2) providing a substitution string for the material in `sub-material` (requires `source-chain` and `output-file`).
        (3, recommended) providing a gdml file `gdml-file` and name of a logical volume `part` from which to take the material information (requires `source-chain` and `output-file`).""",
    )
    alpha_n_parser.add_argument(
        "--output-file", "-o", required=True, type=str, help="Path to output vtx file."
    )
    alpha_n_parser.add_argument(
        "--input-file-sag4n",
        type=str,
        default="",
        help="Path to valid SaG4n input file.",
    )
    alpha_n_parser.add_argument(
        "--sub-material",
        type=str,
        default="",
        help="String to substitute into the template SaG4n input file defining the target material.",
    )
    alpha_n_parser.add_argument(
        "--gdml-file",
        type=str,
        default="",
        help="Path to gdml file to read the material of `part`.",
    )
    alpha_n_parser.add_argument(
        "--part",
        type=str,
        default="",
        help="Part to extract the material information in `gdml-file` from.",
    )
    alpha_n_parser.add_argument(
        "--source-chain",
        "-c",
        type=str,
        default="",
        help="Name of the radiogenic chain to be used as alpha source. Options are [Th232, U238_lower, U238_upper]",
    )
    alpha_n_parser.add_argument(
        "--n-events",
        "-n",
        type=int,
        default=10000000,
        help="Number of primary alphas to simulate. Default is 10000000.",
    )
    alpha_n_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Option for users to use custom seeds. If not set, a random seed will be used.",
    )
    alpha_n_parser.add_argument(
        "--container-runtime",
        type=str,
        default="",
        help="Container runtime for SaG4n ('docker' or 'shifter'). If omitted, runtime is auto-detected.",
    )
    alpha_n_parser.add_argument(
        "--container-image",
        type=str,
        default="moritzneuberger/sag4n-for-revertex:latest",
        help="Container image used to run SaG4n.",
    )
    alpha_n_parser.add_argument(
        "--output-file-sag4n",
        type=str,
        default="",
        help="Folder and stem path of SaG4n output files (.out, .root, .log). These are usually temporary files deleted after processing.",
    )
    musun_gs_parser = subparsers.add_parser(
        "musun-gs",
        help="Generate atmospheric muon kinematics using musun-gs.",
    )
    musun_gs_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output LH5 file.",
    )
    musun_gs_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Total number of muons to generate.",
    )
    musun_gs_parser.add_argument(
        "--default-dimensions",
        type=str,
        default="original",
        help="Use default dimensions for the sampling cuboid. Options are ["
        + ", ".join(musun_gs.DEFAULT_DIMENSIONS.keys())
        + ", custom] (default: original). If 'custom' is selected, the user must provide --dx-cm, --dy-cm, --dz-cm, center_x-cm, center_y-cm, center_z-cm to specify the dimensions of the sampling cuboid.",
    )
    musun_gs_parser.add_argument(
        "--dx-cm",
        type=float,
        help="Full width of the sampling cuboid in x [cm] (original: 4000 = 40 m).",
    )
    musun_gs_parser.add_argument(
        "--dy-cm",
        type=float,
        help="Full width of the sampling cuboid in y [cm] (original: 2000 = 20 m).",
    )
    musun_gs_parser.add_argument(
        "--dz-cm",
        type=float,
        help="Full height of the sampling cuboid in z [cm] (original: 3500 = 35 m).",
    )
    musun_gs_parser.add_argument(
        "--center-x-cm",
        type=float,
        help="x-coordinate of the cuboid center [cm] (original: 0).",
    )
    musun_gs_parser.add_argument(
        "--center-y-cm",
        type=float,
        default=0.0,
        help="y-coordinate of the cuboid center [cm] (original: 0).",
    )
    musun_gs_parser.add_argument(
        "--center-z-cm",
        type=float,
        help="z-coordinate of the cuboid center [cm] (original: 0).",
    )
    musun_gs_parser.add_argument(
        "--container-image",
        type=str,
        default=musun_gs.DEFAULT_CONTAINER_IMAGE,
        help="Docker/Shifter image to use (default: %(default)s).",
    )
    musun_gs_parser.add_argument(
        "--container-runtime",
        type=str,
        default=None,
        help="Container runtime: 'docker' or 'shifter' (default: auto-detect).",
    )

    axion_parser = subparsers.add_parser(
        "axion-kin",
        help="Axion interaction to simulate: axioelectric (e-), compton (e- + gamma), "
"pair (e- + e+), primakoff (gamma)",
    )
    axion_parser.add_argument(
        "--process",
        "-p",
        required=True,
        type=str,
        choices=list(axion.PROCESSES),
        help="Axion interaction to simulate",
    )
    axion_parser.add_argument(
        "--energy",
        "-E",
        required=True,
        type=float,
        help="Total energy of the axion",
    )
    axion_parser.add_argument(
        "--mass",
        "-m",
        default=0.0,
        type=float,
        help="Mass of the axion (default: 0)",
    )
    axion_parser.add_argument(
        "--eunit",
        "-e",
        default="keV",
        type=str,
        help="Unit for the energy and mass (default: keV)",
    )
    axion_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    axion_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )

    args = parser.parse_args(args)

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    if args.command == "beta-kin":
        msg = f"Generating beta kinematics from {args.input_file} to {args.out_file} and seed {args.seed}"
        log.info(msg)

        beta.save_beta_spectrum(
            n_gen=args.n_events,
            out_file=args.out_file,
            in_file=args.input_file,
            seed=args.seed,
            eunit=args.eunit,
        )

    elif args.command == "axion-kin":
        msg = "Generating axion interactions for \n"
        msg += f"process:   {args.process} \n"
        msg += f"energy:    {args.energy} {args.eunit} \n"
        msg += f"mass:      {args.mass} {args.eunit} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed}"
        log.info(msg)

        axion.save_axion_kinematics(
            args.n_events,
            args.out_file,
            process=args.process,
            energy=args.energy,
            mass=args.mass,
            seed=args.seed,
            eunit=args.eunit,
        )
    elif args.command == "hpge-surf-pos":
        msg = "Generating points on the HPGes for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors} ({args.surface_type})"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            surface.sample_hpge_surface,
            hpges=hpges,
            positions=pos,
            surface_type=args.surface_type,
        )
    elif args.command == "hpge-shell-pos":
        msg = "Generating points on the HPGes shells for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors} ({args.surface_type})"
        msg += f"radius : {args.radius}"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            shell.sample_hpge_shell,
            hpges=hpges,
            positions=pos,
            distance=args.radius,
            surface_type=args.surface_type,
        )
    elif args.command == "musun-gs":
        msg = "Generating musun-gs muon kinematics and vertices to %s with seed %s"
        log.info(msg, args.out_file, args.seed)

        if (
            args.default_dimensions != "custom"
            and args.default_dimensions in musun_gs.DEFAULT_DIMENSIONS
        ):
            musun_gs.generate_musun_primaries(
                n_muons=args.n_events,
                out_file=args.out_file,
                seed=args.seed,
                default_dimensions=args.default_dimensions,
                container_image=args.container_image,
                container_runtime=args.container_runtime,
            )
        elif args.default_dimensions == "custom":
            musun_gs.generate_musun_primaries(
                n_muons=args.n_events,
                out_file=args.out_file,
                seed=args.seed,
                dx_cm=args.dx_cm,
                dy_cm=args.dy_cm,
                dz_cm=args.dz_cm,
                center_x_cm=args.center_x_cm,
                center_y_cm=args.center_y_cm,
                center_z_cm=args.center_z_cm,
                container_image=args.container_image,
                container_runtime=args.container_runtime,
            )
        else:
            msg = f"Invalid value for --default-dimensions: {args.default_dimensions}. Valid options are: {', '.join(musun_gs.DEFAULT_DIMENSIONS.keys())}, custom."
            raise ValueError(msg)

    elif args.command == "hpge-borehole-pos":
        msg = "Generating points on the HPGes boreholes for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors}"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            borehole.sample_hpge_borehole,
            hpges=hpges,
            positions=pos,
        )
    elif args.command == "alpha-n-kin":
        if args.seed == -1:
            random.seed()
            args.seed = random.randint(0, 1000000)

        input_data = {
            "output_file": args.output_file,
            "n_events": args.n_events,
            "seed": args.seed,
            "container_image": args.container_image,
        }

        pathway_sag4n = args.input_file_sag4n != ""
        pathway_sub = args.sub_material != ""
        pathway_gdml = args.gdml_file != "" or args.part != ""

        if (
            (pathway_sag4n and pathway_sub)
            or (pathway_sag4n and pathway_gdml)
            or (pathway_sub and pathway_gdml)
        ):
            msg = "You can only provide one of the following options to specify the target material for the alpha-n interaction: (1) a valid SaG4n input file in `input-file-sag4n`, (2) a substitution string for the material in `sub-material` (with `source-chain`) or (3) a gdml file `gdml-file` and name of a logical volume `part` from which to take the material information (with `source-chain`)."
            raise RuntimeError(msg)

        if (args.gdml_file != "") != (args.part != ""):
            msg = "The `--gdml-file` and `--part` options must be provided together."
            raise RuntimeError(msg)

        if args.container_runtime != "":
            input_data["container_runtime"] = args.container_runtime

        if args.input_file_sag4n != "":
            input_data["input_file_sag4n"] = args.input_file_sag4n
        elif args.sub_material != "":
            input_data["sub_material"] = args.sub_material
            if args.source_chain == "":
                msg = "When using the 'sub-material' option, you also need to provide a 'source-chain' (with --source-chain) to specify the radiogenic chain to be used as alpha source."
                raise RuntimeError(msg)
            input_data["source_chain"] = args.source_chain
        elif args.gdml_file != "" and args.part != "":
            input_data["gdml_file"] = args.gdml_file
            input_data["part"] = args.part
            if args.source_chain == "":
                msg = "When using the 'gdml-file' and 'part' options, you also need to provide a 'source-chain' (with --source-chain) to specify the radiogenic chain to be used as alpha source."
                raise RuntimeError(msg)
            input_data["source_chain"] = args.source_chain
        else:
            msg = "For the alpha-n-kin option, you need to provide either a valid SaG4n input file in `input-file-sag4n`, or a substitution string for the material in `sub-material` (with `source-chain`) or a gdml file `gdml-file` and name of a logical volume `part` from which to take the material information (with `source-chain`)."
            raise RuntimeError(msg)

        if args.output_file_sag4n != "":
            input_data["output_file_sag4n"] = args.output_file_sag4n

        msg = "Generating neutrons and gammas using SaG4n: \n"
        if args.gdml_file != "":
            msg += f"gdml:      {args.gdml_file} \n"
        if args.part != "":
            msg += f"part:      {args.part} \n"
        if args.sub_material != "":
            msg += f"sub-material: \n{args.sub_material} \n"
        if args.input_file_sag4n != "":
            msg += f"input-file-sag4n: {args.input_file_sag4n} \n"
        if args.source_chain != "":
            msg += f"source-chain: {args.source_chain} \n"
        msg += f"output_file:    {args.output_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"n_events:      {args.n_events} \n"
        if args.container_runtime != "":
            msg += f"container-runtime: {args.container_runtime} \n"
        if args.container_image != "":
            msg += f"container-image: {args.container_image} \n"
        log.info(msg)

        alpha_n.generate_alpha_n_spectrum(input_data)