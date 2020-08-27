#!/usr/bin/python
import os
import sys
import torch
import speechbrain as sb
import torch.nn.functional as F
from tqdm.contrib import tqdm
from speechbrain.utils.EER import EER
from speechbrain.utils.data_utils import download_file
from speechbrain.nnet.losses import BinaryContrastiveLoss


# Trains embedding model
class EmbeddingBrain(sb.core.Brain):
    def compute_embeddings(self, wavs, lens, init_params=False):
        """Computes the embeddings given a batch of input waveforms.
        """
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        if params.freeze_embeddings:
            params.embedding_model.eval()
            with torch.no_grad():
                emb = params.embedding_model(feats, init_params=init_params)
        else:
            emb = params.embedding_model(feats, init_params=init_params)

        emb = params.mean_var_norm_emb(
            emb, torch.ones(emb.shape[0]).to("cuda:0")
        )
        return emb

    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        wavs, lens = wavs.to(params.device), lens.to(params.device)

        if stage == "train" and hasattr(params, "num_corrupts"):
            wavs_aug_list = []
            for i in range(params.num_corrupts):
                # Addding noise and reverberation
                wavs_aug = params.env_corrupt(wavs, lens, init_params)

                # Adding time-domain augmentation
                wavs_aug = params.augmentation(wavs_aug, lens, init_params)

                wavs_aug_list.append(wavs_aug)
            wavs = torch.cat([wavs] + wavs_aug_list, dim=0)
            lens = lens.repeat_interleave(1 + params.num_corrupts, dim=0)

        emb = self.compute_embeddings(wavs, lens, init_params)
        return emb, lens

    def compute_objectives(self, outputs, targets, stage="train"):
        outputs, lens = outputs
        uttid, spkid, _ = targets

        spkid, lens = spkid.to(params.device), lens.to(params.device)
        if stage == "train" and hasattr(params, "num_corrupts"):
            spkid = spkid.repeat([1 + params.num_corrupts, 1])

        loss = params.cont_wrapper(outputs, spkid)

        stats = {}
        if stage != "train":
            stats["loss"] = loss

        return loss, stats

    def compute_embeddings_loop(self, data_loader):
        """Computes the embeddings of all the waveforms specified in the
        dataloader.
        """
        embedding_dict = {}

        self.modules.eval()
        with torch.no_grad():
            for (batch,) in tqdm(data_loader, dynamic_ncols=True):
                seg_ids, wavs, lens = batch
                wavs, lens = wavs.to(params.device), lens.to(params.device)
                emb = self.compute_embeddings(wavs, lens, init_params=False)
                for i, seg_id in enumerate(seg_ids):
                    embedding_dict[seg_id] = emb[i].detach().clone()
        return embedding_dict

    def compute_EER(self,):
        """ Computes the EER using the standard voxceleb test split
        """
        # Computing  enrollment and test embeddings
        print("Computing enroll/test embeddings...")
        enrol_dict = self.compute_embeddings_loop(enrol_set_loader)
        test_dict = self.compute_embeddings_loop(test_set_loader)

        print("Computing EER..")
        # Reading standard verification split
        gt_file = os.path.join(params.data_folder, "meta", "veri_test.txt")
        samples = []
        labs = []
        positive_scores = []
        negative_scores = []

        for i, line in enumerate(open(gt_file)):
            labs.append(int(line.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = line.split(" ")[1].rstrip().split(".")[0].strip()
            test_id = line.split(" ")[2].rstrip().split(".")[0].strip()
            sample = torch.cat(
                [enrol_dict[enrol_id], test_dict[test_id]], dim=1
            )
            samples.append(sample)

            if i % params.batch_size:
                samples = torch.cat(samples)
                if isinstance(params.cont_loss, BinaryContrastiveLoss):
                    outputs = params.projection(samples)
                    scores = torch.sigmoid(outputs)
                else:
                    enrol, test = torch.chunk(samples, 2, dim=1)
                    scores = F.cosine_similarity(enrol, test).unsqueeze(2)

                for j, score in enumerate(scores.tolist()):
                    if labs[j] == 1:
                        positive_scores.append(score[0])
                    else:
                        negative_scores.append(score[0])
                labs = []
                samples = []
        del enrol_dict, test_dict
        eer = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
        return eer * 100

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        EER = self.compute_EER()
        valid_stats["EER"] = [EER]

        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["loss"]
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        avg_loss = float(sum(valid_stats["loss"]) / len(valid_stats["loss"]))
        params.checkpointer.save_and_keep_only(
            meta={"loss": avg_loss}, min_keys=["loss"]
        )


# Begin Recipe!
if __name__ == "__main__":

    # This hack needed to import data preparation script from ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))
    from voxceleb_prepare import prepare_voxceleb  # noqa E402

    # Load hyperparameters file with command-line overrides
    params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params.output_folder,
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_voxceleb(
        data_folder=params.data_folder,
        save_folder=params.save_folder,
        splits=["train", "dev", "enrol"],
        split_ratio=[90, 10],
        seg_dur=300,
        vad=False,
        rand_seed=params.seed,
        source=params.voxceleb_source
        if hasattr(params, "voxceleb_source")
        else None,
    )

    # Data loaders
    train_set = params.train_loader()
    valid_set = params.valid_loader()
    enrol_set_loader = params.enrol_loader()
    test_set_loader = params.test_loader()

    # Embedding Model
    modules = [params.embedding_model, params.cont_loss]
    first_x, first_y = next(iter(train_set))
    if hasattr(params, "augmentation"):
        modules.append(params.augmentation)

    if isinstance(params.cont_loss, BinaryContrastiveLoss):
        print("Use BinaryContrastiveLoss")

    # Object initialization for training embedding model
    xvect_brain = EmbeddingBrain(
        modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
    )

    # Function for pre-trained model downloads
    def download_and_pretrain():
        """ Downloads the specified pre-trained model
        """
        save_model_path = params.output_folder + "/save/embedding_model.ckpt"
        if "http" in params.embedding_file:
            download_file(params.embedding_file, save_model_path)
        params.embedding_model.load_state_dict(
            torch.load(save_model_path), strict=True
        )

    if hasattr(params, "embedding_file"):
        download_and_pretrain()
    # Recover checkpoints
    params.checkpointer.recover_if_possible()

    # Train the Embedding model
    xvect_brain.fit(
        params.epoch_counter, train_set=train_set, valid_set=valid_set,
    )
    print("Embedding model training completed!")
