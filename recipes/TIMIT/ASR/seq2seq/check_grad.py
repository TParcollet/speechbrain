#!/usr/bin/env python3
"""Recipe for doing ASR with phoneme targets and mPER loss on the TIMIT dataset.

To run this recipe, do the following:
> python experiment_mWER.py mWER.yaml --data_folder /path/to/TIMIT

Authors
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Abdel Heba 2020
 * Sung-Lin Yeh 2020
"""
import os
import sys
import speechbrain as sb
import pandas as pd
import torch

from utils.memory import _add_memory_hooks
from utils.plot import plot_mem


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, x, y, stage):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        if stage == sb.Stage.TRAIN:
            """
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                phns = torch.cat([phns, phns])
            """
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats)

        if stage == sb.Stage.VALID:
            # set max decoding step to the label length
            self.hparams.beam_searcher.max_decode_ratio = phns.size(1) / x.size(
                1
            )
            (
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            ) = self.hparams.beam_searcher(x, wav_lens)
            # return p_ctc, p_seq, wav_lens, hyps
            return (
                wav_lens,
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            )

        elif stage == sb.Stage.TEST:
            (
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            ) = self.hparams.beam_searcher(x, wav_lens)
            return (
                wav_lens,
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            )

        elif stage == sb.Stage.TRAIN:
            # set max decoding step to the label length
            self.hparams.sampler.max_decode_ratio = phns.size(1) / x.size(1)
            # n-best hyps for minWER loss
            (
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            ) = self.hparams.sampler(x, wav_lens)
            return (
                wav_lens,
                hyps,
                topk_hyps,
                topk_scores,
                topk_len,
                topk_logprobs,
            )

    def compute_objectives(self, predictions, targets, stage):
        (
            wav_lens,
            hyps,
            topk_hyps,
            topk_scores,
            topk_length,
            topk_logprobs,
        ) = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)

        """
        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            phns = torch.cat([phns, phns], dim=0)
            phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
        """

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        # phns_with_eos = sb.data_io.data_io.append_eos_token(
        #    phns, length=abs_length, eos_index=self.hparams.eos_index
        # )

        # convert to speechbrain-style relative length
        # rel_length = (abs_length + 1) / phns_with_eos.shape[1]

        # loss_seq = self.hparams.seq_cost(
        #    topk_logprobs[:, 0], phns_with_eos, rel_length
        # )
        loss = self.hparams.minPER_cost(
            topk_hyps, phns, topk_length, abs_length, topk_scores,
        )
        # loss = loss_seq

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # self.seq_metrics.append(ids, topk_logprobs[:, 0], phns_with_eos, rel_length)
            # self.mPER_metrics.append(
            #    ids, topk_hyps, phns, topk_length, abs_length, topk_scores
            # )
            self.per_metrics.append(
                ids, hyps, phns, None, phn_lens, self.hparams.ind2lab
            )

        return loss

    def fit_batch(self, batch):
        inputs, targets = batch
        # compute mem
        mem_log = []
        exp = "baseline"

        try:
            exp = exp or f"exp_{len(mem_log)}"
            hr = []
            idx = 0
            for _, module in enumerate(self.hparams.modules):
                for m in self.hparams.modules[module].modules():
                    _add_memory_hooks(idx, m, mem_log, exp, hr)
                    idx += 1
            print(idx)
            try:
                predictions = self.compute_forward(
                    inputs, targets, sb.Stage.TRAIN
                )
                loss = self.compute_objectives(
                    predictions, targets, sb.Stage.TRAIN
                )
                loss.backward()
            finally:
                [h.remove() for h in hr]
        except Exception as e:
            print(f"log_mem failed because of {e}")
        # mem_log.extend(mem_log)

        df = pd.DataFrame(mem_log)
        df.to_csv("mem_log_mwer.csv")
        plot_mem(
            df,
            exps=["baseline"],
            output_file=f"baseline_memory_plot_1.png",
            normalize_call_idx=False,
        )
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss = self.compute_objectives(predictions, targets, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        self.mPER_metrics = self.hparams.mPER_stats()
        self.seq_metrics = self.hparams.seq_stats()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            if self.root_process:
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch, "lr": old_lr},
                    train_stats={"loss": self.train_loss},
                    valid_stats={
                        "loss": stage_loss,
                        # "mPER_loss": self.mPER_metrics.summarize("average"),
                        "PER": per,
                    },
                )
                self.checkpointer.save_and_keep_only(
                    meta={"PER": per}, min_keys=["PER"]
                )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "PER": per},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("mPER loss stats:\n")
                self.mPER_metrics.write_stats(w)
                # w.write("\nseq2seq loss stats:\n")
                # self.seq_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "mPER loss and PER stats written to file",
                    self.hparams.wer_file,
                )


if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from timit_prepare import prepare_timit  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.load_extended_yaml(fin, overrides)
    # TODO
    # download spbrain seq2seq TIMIT model
    # for fine-tunning with mPER Loss

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_timit(
        data_folder=hparams["data_folder"],
        splits=["train", "dev", "test"],
        save_folder=hparams["data_folder"],
    )

    # Collect index to label conversion dict for decoding
    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["phn"]["index2lab"]

    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(asr_brain.hparams.epoch_counter, train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), min_key="PER")
