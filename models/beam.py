
import torch


class Beam(object):
    def __init__(self, size, vocab, device, score_scheme='lennorm', n_best=1,
            min_length=0):

        self.size = size

        #self.scores = self.torch.FloatTensor(size).zero_()
        self.scores = torch.zeros(size).to(device).float()

        self.back_pointers = []

        #self.tokens = [self.torch.LongTensor(size).fill_(vocab['<PAD>'])
        self.tokens = [torch.ones(size).to(device).long() * vocab['<PAD>']]
        self.tokens[0][0] = vocab['<']

        self.eos_top = False

        self.finished = []
        self.n_best = n_best

        if score_scheme == 'gnmt':
            self.global_scorer = GNMTGlobalScorer(alpha=0.2, beta=0.2)
        elif score_scheme == 'lennorm':
            self.global_scorer = LengthNormScorer()
        else:
            self.global_scorer = None
        self.global_state = {}

        self.min_length = min_length

        self.EOS = vocab['>']
        self.vocab = vocab

    def get_last_token(self):
        return self.tokens[-1]

    def get_last_pointer(self):
        return self.back_pointers[-1]

    def advance(self, word_probs, debug=False):
        num_words = word_probs.size(1)

        cur_len = len(self.tokens)
        if cur_len < self.min_length:
            for k in range(word_probs.size(0)):
                word_probs[k][self.EOS] = -1e20

        if len(self.back_pointers) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)

            for i in range(self.tokens[-1].size(0)):
                if self.tokens[-1][i] == self.EOS:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)

        best_scores, best_scores_id = flat_beam_scores.topk(self.size, dim=0,
            largest=True, sorted=True)

        if debug:
            print(best_scores.tolist())
            print(best_scores_id.tolist())
            print(self.vocab['.'], self.vocab.get(best_scores_id.tolist()[0] % num_words))

        self.scores = best_scores

        back_pointer = best_scores_id // num_words

        self.back_pointers.append(back_pointer)
        self.tokens.append(best_scores_id % num_words)

        if debug:
            print('back point', back_pointer.tolist())
            print('token', self.tokens[-1].tolist())


        if self.global_scorer is not None:
            self.global_scorer.update(self)
            scores = self.global_scorer.score(self, self.scores)
        else:
            scores = self.scores

        for k in range(self.tokens[-1].size(0)):
            if self.tokens[-1][k] == self.EOS:
                self.finished.append((scores[k], len(self.tokens) - 1, k))

        if debug:
            print('adfasd', len(self.finished))

        self.eos_top = (self.tokens[-1][0] == self.EOS)

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=1):
        if minimum is not None:
            scores = self.scores
            if self.global_scorer is not None:
                scores = self.global_scorer.score(self, self.scores)
            k = 0
            while k < len(self.tokens[-1]) and len(self.finished) < minimum:
                if self.tokens[-1][k] != self.EOS:
                    self.finished.append((scores[k], len(self.tokens) - 1, k))
                k += 1
        self.finished.sort(key=lambda a: -a[0])
        return zip(*self.finished)

    def get_seq(self, timestep, pos):
        seq = []
        for step in range(len(self.back_pointers[:timestep]) - 1, -1, -1):
            seq.append(self.vocab.get(self.tokens[step + 1][pos].item()))
            pos = self.back_pointers[step][pos]
        seq.append(self.vocab.get(self.tokens[0][pos].item()))
        return seq[::-1]


class GNMTGlobalScorer(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.clamp(cov, max=1).log().sum(dim=1)
        l_term = (((5 + len(beam.tokens)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update(self, beam):
        if len(beam.back_pointers) == 1:
            beam.global_state["coverage"] = beam.attentions[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.back_pointers[-1]).add(beam.attentions[-1])


class LengthNormScorer(object):

    def score(self, beam, logprobs):
        return logprobs / len(beam.tokens)

    def update(self, unused_beam):
        pass
