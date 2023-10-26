import os
import torch
import cv2
from torchvision import transforms
from checkpoint import default_checkpoint, load_checkpoint
from model_attention import Encoder, Decoder
from dataset import START, SPECIAL_TOKENS, load_vocab_market
import numpy as np

# These are not counted as symbol, because they are used for formatting / grouping, and
# they do not render anything on their own.
use_cuda = torch.cuda.is_available()


class Dense_net():
    def __init__(self, token_path=None, checkpoint_path=None, gpu_index=-1, beam_width=1):
        torch.cuda.set_device(gpu_index)
        self.device = torch.device("cuda")
        self.beam_width = beam_width
        self.img_width = 256
        self.img_height = 64
        self.input_size = (self.img_height, self.img_width)
        self.low_res_shape = (684, self.input_size[0] // 16, self.input_size[1] // 16)
        self.high_res_shape = (792, self.input_size[0] // 8, self.input_size[1] // 8)
        self.token_to_id, self.id_to_token = load_vocab_market(token_path)
        self.checkpoint_name, _ = os.path.splitext(os.path.basename(checkpoint_path))
        self.checkpoint = (
            load_checkpoint(checkpoint_path, gpu_index=gpu_index, cuda=True)
            if checkpoint_path
            else default_checkpoint
        )
        self.encoder_checkpoint = self.checkpoint["model"].get("encoder")
        self.decoder_checkpoint = self.checkpoint["model"].get("decoder")

        self.enc = Encoder(img_channels=3, checkpoint=self.encoder_checkpoint).to(self.device)
        # para = sum([np.prod(list(p.size())) for p in self.enc.parameters()])
        # print('Model {} : params: {:4f}M'.format(self.enc._get_name(), para * 4 / 1000 / 1000))
        self.dec = Decoder(len(self.id_to_token), self.low_res_shape, self.high_res_shape,
                           checkpoint=self.decoder_checkpoint,
                           device=self.device, ).to(self.device)
        # para = sum([np.prod(list(p.size())) for p in self.dec.parameters()])
        # print('Model {} : params: {:4f}M'.format(self.dec._get_name(), para * 4 / 1000 / 1000))
        self.enc.eval()
        self.dec.eval()

        # strip_only means that only special tokens on the sides are removed. Equivalent to
        # String.strip()

    def remove_special_tokens(self, tokens, special_tokens=SPECIAL_TOKENS, strip_only=False):
        if strip_only:
            num_left = 0
            num_right = 0
            for tok in tokens:
                if tok not in special_tokens:
                    break
                num_left += 1
            for tok in reversed(tokens):
                if tok not in special_tokens:
                    break
                num_right += 1
            return tokens[num_left:-num_right]
        else:
            return torch.tensor([tok for tok in tokens if tok not in special_tokens], dtype=tokens.dtype)

    def decode(self, sub_sequence, id_to_token):
        res = ''
        for index in sub_sequence:
            res += id_to_token[index.item()]
        return res

    # Convert hypothesis batches to hypothesis grouped by sequence.
    def unbatch_hypotheses(self, hypotheses):
        if not hypotheses:
            return []
        hypotheses_by_seq = [[] for _ in hypotheses[0]["probability"]]
        for h in hypotheses:
            for i in range(len(h["probability"])):
                single_h = {
                    "sequence": {"full": h["sequence"]["full"][i]},
                    # The hidden weights have batch size in the second dimension, not first.
                    "hidden": h["hidden"][:, i],
                    "attn": {"low": h["attn"]["low"][i], "high": h["attn"]["high"][i]},
                    "probability": h["probability"][i],
                }
                hypotheses_by_seq[i].append(single_h)
        return hypotheses_by_seq

    def batch_single_hypotheses(self, single_hypotheses):
        # It might be possible that the different sequences have a different number of total
        # hypotheses, since there might be duplicates in one of them. To prevent that take
        # the lowest number that is available. It might be smaller than the beam width.
        # But there can only be batches where each sequence is present.
        min_len = min(len(hs) for hs in single_hypotheses)
        batched_hypotheses = []
        for i in range(min_len):
            batch_h = {
                "sequence": {
                    "full": torch.stack(
                        [hs[i]["sequence"]["full"] for hs in single_hypotheses]
                    )
                },
                # The hidden weights have batch size in the second dimension, not first.
                "hidden": torch.stack([hs[i]["hidden"] for hs in single_hypotheses], dim=1),
                "attn": {
                    "low": torch.stack([hs[i]["attn"]["low"] for hs in single_hypotheses]),
                    "high": torch.stack(
                        [hs[i]["attn"]["high"] for hs in single_hypotheses]
                    ),
                },
                "probability": torch.stack(
                    [hs[i]["probability"] for hs in single_hypotheses]
                ),
            }
            batched_hypotheses.append(batch_h)
        return batched_hypotheses

    # Picks the k sequences with the best probabilities. Each sequence is inspected
    # separately and at the end new hypotheses are created by stacking the k best ones of
    # each sequence to create batches, that can be used for the next step.
    def pick_top_k_unique(self, hypotheses, count):
        sorted_hypotheses = [
            sorted(hs, key=lambda h: h["probability"].item(), reverse=True)
            for hs in self.unbatch_hypotheses(hypotheses)
        ]
        unique_hypotheses = [[] for _ in sorted_hypotheses]

        for i, hs in enumerate(sorted_hypotheses):
            for h in hs:
                if len(unique_hypotheses[i]) >= count:
                    break
                already_exists = False
                for h_uniq in unique_hypotheses[i]:
                    already_exists = torch.equal(
                        h["sequence"]["full"], h_uniq["sequence"]["full"]
                    )
                    if already_exists:
                        break
                if not already_exists:
                    unique_hypotheses[i].append(h)

        return self.batch_single_hypotheses(unique_hypotheses)

    def recognize(self, input_data):
        special_tokens = [self.token_to_id[tok] for tok in SPECIAL_TOKENS]
        input_data = torch.from_numpy(input_data).to(self.device)
        # if type(input_data).__name__ == 'ndarray':
        #     loader = transforms.Compose([transforms.ToTensor()])
        #     # input_data = cv2.resize(input_data, (self.img_width, self.img_height))
        # elif type(input_data).__name__ == 'Image':
        #     loader = transforms.Compose([
        #         transforms.Resize((self.img_height, self.img_width)),
        #         transforms.ToTensor()])
        # else:
        #     raise ValueError('输入数据格式异常')
        #
        # input_data = loader(input_data).unsqueeze(0).to(self.device)
        # The last batch may not be a full batch
        curr_batch_size = len(input_data)

        batch_max_len = 40

        enc_low_res, enc_high_res = self.enc(input_data)
        # Decoder needs to be reset, because the coverage attention (alpha)
        # only applies to the current image.
        self.dec.reset(curr_batch_size)
        hidden = self.dec.init_hidden(curr_batch_size).to(self.device)
        # Starts with a START token
        sequence = torch.full(
            (curr_batch_size, 1),
            self.token_to_id[START],
            dtype=torch.long,
            device=self.device,
        )
        hypotheses = [
            {
                "sequence": {"full": sequence},
                "hidden": hidden,
                "attn": {
                    "low": self.dec.coverage_attn_low.alpha,
                    "high": self.dec.coverage_attn_high.alpha,
                },
                # This will be a tensor of probabilities (one for each batch), but at
                # the beginning it can be 1.0 because it will be broadcast for the
                # multiplication and it means the first tensor of probabilities will be
                # kept as is.
                "probability": 1.0,
            }
        ]
        for i in range(batch_max_len - 1):
            step_hypotheses = []
            for hypothesis in hypotheses:
                curr_sequence = hypothesis["sequence"]["full"]
                previous = curr_sequence[:, -1].view(-1, 1)
                curr_hidden = hypothesis["hidden"]
                # Set the attention to the corresponding values, otherwise it would use
                # the attention from another hypothesis.
                self.dec.coverage_attn_low.alpha = hypothesis["attn"]["low"]
                self.dec.coverage_attn_high.alpha = hypothesis["attn"]["high"]
                out, next_hidden = self.dec(previous, curr_hidden, enc_low_res, enc_high_res)
                probabilities = torch.softmax(out, dim=1)
                topk_probs, topk_ids = torch.topk(probabilities, self.beam_width)
                # topks are transposed, because the columns are needed, not the rows.
                # One column is the top values for the batches, and there are k rows.
                for top_prob, top_id in zip(topk_probs.t(), topk_ids.t()):
                    next_sequence = torch.cat(
                        (curr_sequence, top_id.view(-1, 1)), dim=1
                    )
                    probability = hypothesis["probability"] * top_prob
                    next_hypothesis = {
                        "sequence": {"full": next_sequence},
                        "hidden": next_hidden,
                        "attn": {
                            "low": self.dec.coverage_attn_low.alpha,
                            "high": self.dec.coverage_attn_high.alpha,
                        },
                        "probability": probability,
                    }
                    step_hypotheses.append(next_hypothesis)
            # Only the beam_width number of hypotheses with the highest probabilities
            # are kept for the next iteration.
            hypotheses = self.pick_top_k_unique(step_hypotheses, self.beam_width)

        result = []
        for hypothesis in hypotheses:
            sequence = hypothesis["sequence"]
            sequence["removed"] = [
                self.remove_special_tokens(seq, special_tokens) for seq in sequence["full"]
            ]
            for i in sequence['removed']:
                result.append(self.decode(i, self.id_to_token))
        return result


if __name__ == "__main__":
    D = Dense_net(token_path='q1_torch_slim.txt', checkpoint_path='q1_lr.pth', gpu_index=3,
                  beam_width=1)
    image_path = '/data/data1/monky_data_line/wx_slim/z/q1_3-1388_17.jpg'
    batch = []
    input = cv2.imread(image_path)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = cv2.resize(input, (256, 64))
    input = np.array(input, dtype=np.float32)
    input = input / 255
    input = input.transpose(2,0,1)
    # input = np.array(input, dtype=np.float32)
    batch.append(input)
    batch.append(input)
    batch = np.array(batch, dtype=np.float32)

    for res in D.recognize(batch):
        print(res)
