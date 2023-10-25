#%%




# pr = osf.project('cr5mv')
#%%
# # getting help
# $ osf -h
# $ osf <command> -h

# # list all files for a public project
# $ osf -p <projectid> list

# # setup a local folder for an existing project
# $ osf init

# # list all files for a private project
# # set $OSF_PASSWORD to provide the password
# $ osf -p <projectid> -u yourOSFacount@example.com list

# # fetch all files from a project and store them in `output_directory`
# $ osf -p <projectid> clone [output_directory]

# # create a new file in an OSF project
# $ osf -p <projectid> -u yourOSFacount@example.com upload local/file.txt remote/path.txt

# # download a single file from an OSF project
# $ osf -p <projectid> fetch remote/path.txt local/file.txt

# # upload a single file to an OSF project
# $ osf -p <projectid> upload local/path.txt remote/file.txt

# # remove a single file from an OSF project
# $ osf -p <projectid> remove remote/file.txt
#%%


# config_ = dict()
# config_['username'] = 'sromeropinto@g.harvard.edu'
# config_['project'] = 'cr5mv'

# cc = cli.config_from_env(config_)
# cc= cli.init()
# cli.clone()

#%%
# cc = osfclient.OSF(username='sromeropinto@g.harvard.edu')
# # dir(cc)
# cp = cc.project('cr5mv')
# ss=cp.session
#%%
import osfclient
from osfclient import cli
from osfclient.utils import  makedirs, checksum
from tqdm import tqdm
import os
config_=cli.config_from_file()
config_['output'] = 'test_osf'
def clone_py(args):
    """Copy all files from all storages of a project.

    The output directory defaults to the current directory.

    If the project is private you need to specify a username.

    If args.update is True, overwrite any existing local files only if local and
    remote files differ.
    """
    # osf = _setup_osf(args)
    osf = osfclient.OSF()
    project = osf.project(args['project'])
    output_dir = args['output']
    # output_dir = args.project
    # if args.output is not None:
        

    with tqdm(unit='files') as pbar:
        for store in project.storages:
            prefix = os.path.join(output_dir, store.name)

            for file_ in store.files:
                path = file_.path
                if path.startswith('/'):
                    path = path[1:]

                path = os.path.join(prefix, path)
                if os.path.exists(path) and args.update:
                    if checksum(path) == file_.hashes.get('md5'):
                        continue
                directory, _ = os.path.split(path)
                makedirs(directory, exist_ok=True)

                with open(path, "wb") as f:
                    file_.write_to(f)

                pbar.update()

clone_py(config_)