import sys
import toml
from skbuild import setup

# ----------------------------------------------------------------------
# QuEST compilation settings
# ----------------------------------------------------------------------
quest_config = {
    'precision': 2,            # Size of a float; 1 (single),
                               # 2 (double), or 4 (quad precision).
    'multithreaded': True,     # Enable multithreading via OpenMP.
    'distributed': False,      # Enable distributed code via MPI.
    'gpu_accelerated': False,  # Enable Nvidia GPU support via CUDA.
    # CUDA needs to know the compute capability of your GPU. Find your
    # GPU model on https://developer.nvidia.com/cuda-gpus and set
    # 'gpu_compute_capability' to the "Compute Capability" listed
    # there, without the period (e.g. 30 for 3.0).
    'gpu_compute_capability': None,
}
# ----------------------------------------------------------------------


def load_setup_args_from_pyproject():
    """Read and parse project metadata from pyproject.toml.

    This method reads metadata from the `pyproject.toml` file in the
    current directory, extracts project metadata as specified in
    PEP 621, and returns a dictionary whose key-value pairs can be
    passed to `skbuild.setup()`. It is by no means an implementation of
    a full PEP 621 parser and only functions as a temporary bridge
    between `pyproject.toml` and `skbuild.setup()` until scikit-build
    supports reading `pyproject.toml` directly. It makes several
    assumptions that are true for this project at the time of writing,
    but are not required by PEP 621. Therefore it must be checked
    carefully whenever substantial changes to `pyproject.toml` are made.
    """
    setup_args = {}
    pyproject = toml.load('pyproject.toml')
    project = pyproject['project']
    build_system = pyproject['build-system']

    # The arguments we can just copy over verbatim. We don't do any
    # validation or checks for mandatory fields.
    ARGUMENT_MAP = {
        'name': 'name', 'version': 'version', 'description': 'description',
        'requires-python': 'python_requires',
        'dependencies': 'install_requires', 'keywords': 'keywords',
        'classifiers': 'classifiers',
    }
    setup_args = {}
    for pyproj_arg, setup_arg in ARGUMENT_MAP.items():
        if pyproj_arg in project:
            setup_args[setup_arg] = project[pyproj_arg]

    # Put the name of the homepage into the 'url' argument, because
    # PEP 345 specifies 'Home-page' separately from the 'Project-URL'
    # attribute. Our toolchain puts the 'url' argument into the
    # 'Home-page' metadata field, and the 'project_urls' argument into
    # the 'Project_URL' field.
    if 'urls' in project:
        for url_name in project['urls']:
            if url_name.replace("-", "").lower() == "homepage":
                setup_args['url'] = project['urls'][url_name]
                del project['urls'][url_name]
                break  # Only the first in case there are more matches.
        if len(project['urls']) > 0:
            setup_args['project_urls'] = project['urls']

    # We do not put the readme straight into pyproject.toml (even though
    # PEP 621 would allow it), so the parser for the readme section only
    # supports names of .md and .rst files whose content is copied to
    # the long_description argument.
    if 'readme' in project:
        with open(project['readme'], 'r') as readme_file:
            setup_args['long_description'] = readme_file.read()
        if project['readme'].lower().endswith('.md'):
            setup_args['long_description_content_type'] = 'text/markdown'
        elif project['readme'].lower().endswith('.rst'):
            setup_args['long_description_content_type'] = 'text/x-rst'
        else:
            raise ValueError("readme filename must end in .md or .rst")

    if ('license' in project
        and ('file' in project['license']
             or 'text' in project['license'])):
        if ('file' in project['license']
                and 'text' in project['license']):
            raise ValueError("'license' can only specify one of 'file' "
                             "or 'text'.")
        if 'file' in project['license']:
            setup_args['license_files'] = [project['license']['file']]
        else:
            setup_args['license'] = project['license']['text']

    # We deviate slightly from PEP 621 and put the list of authors in
    # the `Author` field, regardless of whether `email` is specified.
    # In addition, as should be the case according to PEP 621, we put
    # a list of names and email addresses in the Author-email field.
    if 'authors' in project:
        proj_authors = ''
        proj_emails = ''
        for author in project['authors']:
            if 'name' in author:
                proj_authors += author['name'] + ", "
                if 'email' in author:
                    proj_emails += (
                        f"{author['name']} <{author['email']}>, ")
            elif 'email' in author:
                proj_emails += author['email'] + ", "
        if proj_authors:
            setup_args['author'] = proj_authors.rstrip(", ")
        if proj_emails:
            setup_args['author_email'] = proj_emails.rstrip(", ")

    # The same as for `authors` also applies to `maintainers`.
    if 'maintainers' in project:
        proj_maintainers = ''
        proj_maint_emails = ''
        for maintainer in project['maintainers']:
            if 'name' in maintainer:
                proj_maintainers += maintainer['name'] + ", "
                if 'email' in maintainer:
                    proj_maint_emails += (
                        f"{maintainer['name']} <{maintainer['email']}>, ")
            elif 'email' in maintainer:
                proj_maint_emails += maintainer['email'] + ", "
        if proj_maintainers:
            setup_args['maintainer'] = proj_maintainers.rstrip(", ")
        if proj_maint_emails:
            setup_args['maintainer_email'] = proj_maint_emails.rstrip(", ")

    entry_points = {}
    if 'scripts' in project:
        entry_points['console_scripts'] = []
        for entry_pt, obj in project['scripts'].items():
            entry_points['console_scripts'] += [f"{entry_pt} = {obj}"]
    if 'gui-scripts' in project:
        entry_points['gui_scripts'] = []
        for entry_pt, obj in project['gui-scripts'].items():
            entry_points['gui_scripts'] += [f"{entry_pt} = {obj}"]
    if 'entry-points' in project:
        for entry_pt_grp_name, entry_pt_grp in project['entry-points'].items():
            entry_points[entry_pt_grp_name] = []
            for entry_pt, obj in entry_pt_grp.items():
                entry_points[entry_pt_grp_name] += [f"{entry_pt} = {obj}"]
    if entry_points:
        setup_args['entry_points'] = entry_points

    # We only ship one package, and it has the project's name. So we
    # just re-use that for the `packages` argument.
    setup_args['packages'] = [project['name']]

    # This should not be necessary because we use PEP 518, but on some
    # systems (specifically Google Colab) setting up the isolated build
    # system by pip is not working correctly. A new cmake version is
    # installed in the build environment, but the old (globally
    # installed) version is called by scikit-build. We therefore also
    # pass the required cmake version to `skbuild.setup()` via the
    # `setup_requires` argument.
    setup_args['setup_requires'] = [package
                                    for package in build_system['requires']
                                    if package.startswith('cmake')]

    return setup_args


def validate_quest_config(quest_config):
    """Validate QuEST compilation configuration.

    The build parameters are re-checked by cmake when building QuEST,
    but raising appropriate exceptions here makes them easier to spot
    and fix for the user.
    """
    if quest_config['gpu_accelerated']:
        if quest_config['precision'] == 4:
            raise ValueError("Quad precision ('precision': 4) not supported on"
                             "CUDA devices. Use lower precision or set 'cuda' "
                             "to False.")
        if quest_config['gpu_compute_capability'] is None:
            raise TypeError("When compiling with GPU support, "
                            "'cuda_compute_capability' must be set.")
        if quest_config['multithreaded']:
            raise ValueError("GPU acceleration and multithreading cannot be "
                             "used at the same time. Disable one of "
                             "'gpu_accelerated' and 'multithreaded'.")
    if quest_config['precision'] not in [1, 2, 4]:
        raise ValueError("Precision must be either 1 (single), 2 (double), "
                         "or 4 (quad precision)")


def set_default_skbuild_arg(arg, val):
    """Set a default value for a given skbuild argument.

    Check all arguments in `sys.argv` *before the first `--`* (which is
    the terminator for skbuild-arguments, later arguments are passed to
    cmake or the build tool) for the argument `arg`. If it is set, do
    nothing and return. If it is not set, set it at the end of the
    argument list to `val`, with `arg=val`.
    """
    insert_at = len(sys.argv)
    for k, cur_arg in enumerate(sys.argv):
        if cur_arg.startswith(arg):
            insert_at = None
            break
        # '--' marks the end of setuptools-arguments, so we must insert
        # before this separator.
        if cur_arg == '--':
            insert_at = k
            break
    if insert_at is not None:
        sys.argv.insert(insert_at, f'{arg}={val}')


def quest_config_to_cmake_args(quest_config):
    """Translate the `quest_config` dictionary to `cmake` arguments.

    The returned list of strings can be passed to `skbuild.setup()` as
    argument `cmake_args` to build QuEST with the chosen configuration.
    """
    quest_cmake_args = [
        "-DPRECISION:STRING=" + str(quest_config['precision']),
        "-DMULTITHREADED:BOOL=" + ("ON" if quest_config['multithreaded']
                                   else "OFF"),
        "-DDISTRIBUTED:BOOL=" + ("ON" if quest_config['distributed']
                                 else "OFF"),
        "-DGPUACCELERATED:BOOL=" + ("ON" if quest_config['gpu_accelerated']
                                    else "OFF")]
    if quest_config['gpu_accelerated']:
        quest_cmake_args.append(
            "-DGPU_COMPUTE_CAPABILITY:STRING="
            + str(quest_config['gpu_compute_capability']))
    return quest_cmake_args


# `scikit-build` disables docstrings for `Release` and `MinSizeRel`
# builds. To prevent this (in a hacky way), set the default build type
# to a new (made up) value `RelWithDocs`. This does not override a
# user-specified build type.
set_default_skbuild_arg("--build-type", "RelWithDocs")

validate_quest_config(quest_config)
quest_cmake_args = quest_config_to_cmake_args(quest_config)
setup_args = load_setup_args_from_pyproject()

setup(
    **setup_args,
    cmake_args=quest_cmake_args,
)
