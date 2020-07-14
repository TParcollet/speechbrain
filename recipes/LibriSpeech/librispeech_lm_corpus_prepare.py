"""
Data preparation for LM corpus.

Author
------
Ju-Chieh Chou 2020
"""

import os
import csv
import logging
from speechbrain.data_io.data_io import (
    load_pkl,
    save_pkl,
)
import h5py

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_lm_corpus_prepare.pkl"


def prepare_librispeech_lm_corpus(
    data_folder, save_folder, filename, data_format, select_n_sentences=None
):
    """
    This function prepares the csv file/hdf5 file for the LibriSpeech LM corpus.
    Download link: http://www.openslr.org/11/

    Arguments:
    data_folder : str
        Path to the folder of LM (normalized) corpus.
    save_folder : str
        folder to store the csv file.
    filename : str
        The filename of csv/hdf5 file.
    data_format : string
        The format of prepared file (one of hdf5 / csv).
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.

    Example
    -------
    >>> data_folder = 'dataset/LibriSpeech'
    >>> save_folder = 'librispeech_lm'
    >>> prepare_librispeech(data_folder, save_folder, 'lm_corpus.csv', data_format='csv')
    """
    conf = {
        "select_n_sentences": select_n_sentences,
        "data_format": data_format,
    }
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, filename, conf):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    data_path = os.path.join(data_folder, "librispeech-lm-norm.txt")

    if data_format == "csv":
        create_csv(
            data_path, save_folder, filename, select_n_sentences,
        )
    elif data_format == "hdf5":
        create_hdf5(
            data_path, save_folder, filename, select_n_sentences,
        )
    else:
        raise ValueError(
            f"data_format should be one of (csv / hdf5), " "got {data_format}."
        )

    # saving options
    save_pkl(conf, save_opt)


def skip(save_folder, filename, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    save_folder : str
        The location of the seave directory
    filename : str
        The filename of the file.
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    if not os.path.isfile(os.path.join(save_folder, filename)):
        skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def create_csv(
    data_path, save_folder, filename, select_n_sentences,
):
    """
    Create the csv file.

    Arguments
    ---------
    data_path : str
        The path of LM corpus txt file.
    save_folder : str
        Location of the folder for storing the csv.
    filename : str
        The filename of csv file.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, filename)

    # Preliminary prints
    msg = "\tCreating csv in  %s..." % (csv_file)
    logger.debug(msg)

    # TODO: Using duration as length
    header = [
        "ID",
        "duration",
        "char",
        "char_format",
        "char_opts",
    ]

    snt_cnt = 0
    with open(data_path, "r") as f_in, open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(header)
        for snt_id, line in enumerate(f_in):
            wrds = "_".join(line.strip().split(" "))

            # skip empty sentences
            if len(wrds) == 0:
                continue

            # replace space to <space> token
            chars_lst = [c for c in wrds]
            chars = " ".join(chars_lst)

            # TODO: duration set to char len temporarily
            csv_line = [
                snt_id,
                len(chars_lst),
                str(chars),
                "string",
                "",
            ]
            csv_writer.writerow(csv_line)
            snt_cnt = snt_cnt + 1

            if snt_cnt == select_n_sentences:
                break

    # Final print
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)


def create_hdf5(
    data_path, save_folder, filename, select_n_sentences,
):
    """
    Create the hdf5 file.

    Arguments
    ---------
    data_path : str
        The path of LM corpus txt file.
    save_folder : str
        Location of the folder for storing the csv.
    filename : str
        The filename of hdf5 file.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    hdf5_file = os.path.join(save_folder, filename)

    # Preliminary prints
    msg = "\tCreating hdf5 in  %s..." % (hdf5_file)
    logger.debug(msg)

    snt_cnt = 0
    all_wrds, all_chars = [], []
    with open(data_path, "r") as f_in:
        for snt_id, line in enumerate(f_in):
            wrds = line.strip()
            wrds_lst = wrds.split(" ")

            # skip empty sentences
            if len(wrds) == 0:
                continue

            # replace space to <space> token
            chars_lst = [c for c in "_".join(wrds_lst)]
            chars = " ".join(chars_lst)

            all_wrds.append(wrds)
            all_chars.append(chars)

            snt_cnt = snt_cnt + 1
            if snt_cnt == select_n_sentences:
                break

    with h5py.File(hdf5_file, "w") as f_h5:
        dset = f_h5.create_dataset(
            "wrd", (len(all_wrds),), dtype=h5py.string_dtype()
        )
        dset[:] = all_wrds
        dset = f_h5.create_dataset(
            "char", (len(all_chars),), dtype=h5py.string_dtype()
        )
        dset[:] = all_chars

    # Final print
    msg = "\t%s sucessfully created!" % (hdf5_file)
    logger.debug(msg)
