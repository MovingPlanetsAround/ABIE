import os
import sys
import h5py
import glob
import shutil
import numpy as np
from optparse import OptionParser


parser = OptionParser()
parser.add_option(
    "-f",
    "--file",
    action="store",
    type="string",
    dest="file",
    help="The hdf5 file to be serialized (e.g., p_sys_0.hdf5.unfinished)",
)
parser.add_option(
    "--dataset",
    action="store",
    type="string",
    dest="dataset",
    default=None,
    help="The dataset of interest to be serialized (e.g., mass). If not specified, serialize all datasets.",
)
# parser.add_option('-o', '--output_file', action='store', type='string', dest='output_file',
#                   help='The name of the output file (optional)')
(options, args) = parser.parse_args()
if options.file is None:
    print("Usage: python snapshot_serialization -f <file_to_be_serialized.hdf5>")
    sys.exit(0)

h5fns = glob.glob(options.file)
print(h5fns)
if len(h5fns) > 0:
    for h5fn_id, h5fn in enumerate(h5fns):
        output_file_name = os.path.splitext(os.path.basename(h5fn))[0]
        print(("Processing %s" % h5fn))
        # temporarily copying the .unfinished file to /tmp
        new_file_path = os.path.join("/tmp", os.path.basename(h5fn))
        shutil.copyfile(h5fn, new_file_path)
        with h5py.File(new_file_path, "r") as h5f, h5py.File(
            output_file_name, "w"
        ) as h5f_out:
            step_id_list = []
            step_len_vec = np.zeros(len(h5f), dtype=np.int)
            if len(h5f["/Step#0/hash"]) > 0:
                vec_hash = h5f["/Step#0/hash"].value[
                    0
                ]  # the hash array of the first step (which contains all particles)
            else:
                vec_hash = h5f["/Step#1/hash"].value[
                    0
                ]  # the hash array of the first step (which contains all particles)
            print("vec_hash", vec_hash.shape)
            dset_dict = dict()
            print("Mapping internal data structure...")
            for dset_name in h5f:
                if "Step#" in dset_name:
                    step_id = int(dset_name.split("#")[1])
                    step_id_list.append(step_id)
                    step_len_vec[step_id] = h5f[
                        "/%s/%s" % (dset_name, "x")
                    ].value.shape[0]
                else:
                    # it is a serialized dataset, just copy it to the serialized file
                    h5f_out.create_dataset(dset_name, data=h5f[dset_name].value)

            if len(step_id_list) > 0:  # if more than zero steps
                cursor = 0
                for step_id in sorted(step_id_list):
                    step_name = "Step#%d" % step_id
                    print(step_name)
                    h5g = h5f[step_name]
                    if len(h5g["hash"].value) == 0:
                        continue
                    indices = np.where(np.in1d(vec_hash, h5g["hash"].value))[
                        0
                    ]  # the indices to be updated

                    if options.dataset is not None:
                        dset_list = [options.dataset]
                    else:
                        dset_list = h5g.keys()

                    for dset_name in dset_list:
                        print(dset_name)
                        tmp_data = h5g[dset_name].value
                        lower_idx = cursor
                        higher_idx = cursor + tmp_data.shape[0]
                        print(
                            "cursors",
                            lower_idx,
                            higher_idx,
                            cursor,
                            step_len_vec[step_id],
                        )
                        if dset_name not in dset_dict.keys():
                            # allocate memory
                            if tmp_data.ndim == 2:
                                dset_dict[dset_name] = (
                                    np.zeros((np.sum(step_len_vec), tmp_data.shape[1]))
                                    * np.nan
                                )
                            elif h5g[dset_name].value.ndim == 1:
                                dset_dict[dset_name] = (
                                    np.zeros(np.sum(step_len_vec)) * np.nan
                                )
                        # append the data to the allocated big array
                        # dset_dict[dset_name][step_id*tmp_data.shape[0]:(step_id+1)*tmp_data.shape[0]][indices] = tmp_data
                        if tmp_data.ndim == 2:
                            print(
                                dset_name,
                                tmp_data.shape,
                                dset_dict[dset_name][lower_idx:higher_idx].shape,
                            )
                            print(indices, vec_hash.shape)
                            dset_dict[dset_name][lower_idx:higher_idx, :][
                                :, indices
                            ] = tmp_data
                        else:
                            # print dset_name, tmp_data.shape, dset_dict[dset_name][lower_idx:higher_idx].shape
                            dset_dict[dset_name][lower_idx:higher_idx] = tmp_data
                    cursor += step_len_vec[step_id]
                    # dset_dict[dset_name][step_id][indices] = tmp_data
                for dset_name in dset_dict.keys():
                    if dset_name == "hash":
                        h5f_out.create_dataset(
                            dset_name, data=dset_dict[dset_name], dtype=np.int64
                        )
                    else:
                        h5f_out.create_dataset(dset_name, data=dset_dict[dset_name])
        os.remove(new_file_path)
