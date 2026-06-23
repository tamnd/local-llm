package llama

import (
	"math"
	"math/rand"
	"sort"
)

// This file is the pure-Go core of the engine: token sampling, streaming
// stop-sequence handling, and the speculative-decoding acceptance test. It is
// always compiled and unit-tested on every platform, so the parts of generation
// that are easy to get subtly wrong are exercised without a GPU. The cgo binding
// in cgo.go reads raw logits out of libllama and hands them here; it never makes
// a sampling or acceptance decision itself.

// sampler turns a logits vector into a token id. With temperature at or below
// zero it is greedy (argmax), which is also the mode speculative decoding runs
// in. Above zero it applies top-k, top-p, and min-p filtering then samples from
// the temperature-scaled distribution with a seeded RNG, so a fixed seed gives a
// reproducible stream.
type sampler struct {
	temp float64
	topP float64
	minP float64
	topK int
	rng  *rand.Rand
}

func newSampler(s Sampling) *sampler {
	return &sampler{
		temp: float64(s.Temperature),
		topP: float64(s.TopP),
		minP: float64(s.MinP),
		topK: s.TopK,
		rng:  rand.New(rand.NewSource(int64(s.Seed))),
	}
}

// greedy reports whether this sampler always takes the most likely token. The
// speculative path is only exact in greedy mode, so the engine checks this to
// decide whether a draft model can be used for a given request.
func (s *sampler) greedy() bool { return s.temp <= 0 }

// sample picks a token id from a logits vector. The vector is indexed by token
// id and is consumed read-only.
func (s *sampler) sample(logits []float32) int {
	if s.greedy() {
		return argmax(logits)
	}

	// Build (id, logit) pairs, scale by temperature, and turn into probabilities
	// with a numerically stable softmax. Filtering happens on the sorted list.
	type cand struct {
		id float64
		p  float64
	}
	cands := make([]cand, len(logits))
	for i, l := range logits {
		cands[i] = cand{id: float64(i), p: float64(l) / s.temp}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].p > cands[b].p })

	if s.topK > 0 && s.topK < len(cands) {
		cands = cands[:s.topK]
	}

	maxLogit := cands[0].p
	var sum float64
	for i := range cands {
		cands[i].p = math.Exp(cands[i].p - maxLogit)
		sum += cands[i].p
	}
	for i := range cands {
		cands[i].p /= sum
	}

	// min-p drops everything below a fraction of the top probability, which keeps
	// the tail from leaking low-quality tokens at high temperature.
	if s.minP > 0 {
		cut := s.minP * cands[0].p
		kept := cands[:0]
		for _, c := range cands {
			if c.p >= cut {
				kept = append(kept, c)
			}
		}
		cands = kept
	}

	// top-p keeps the smallest prefix whose cumulative mass reaches topP.
	if s.topP > 0 && s.topP < 1 {
		var cum float64
		for i := range cands {
			cum += cands[i].p
			if cum >= s.topP {
				cands = cands[:i+1]
				break
			}
		}
	}

	// Renormalize the survivors and draw.
	var total float64
	for _, c := range cands {
		total += c.p
	}
	r := s.rng.Float64() * total
	for _, c := range cands {
		r -= c.p
		if r <= 0 {
			return int(c.id)
		}
	}
	return int(cands[len(cands)-1].id)
}

// argmax returns the index of the largest value. It is the greedy decode step
// and the per-position target prediction in speculative verify.
func argmax(xs []float32) int {
	best := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] > xs[best] {
			best = i
		}
	}
	return best
}

// acceptPrefix is the speculative-decoding acceptance test. draft is the tokens
// the draft model proposed; target is what the target model would itself produce
// greedily at each of those positions (its own prediction for the slot the draft
// token fills). The number of leading positions where they agree is the accepted
// run: those draft tokens are correct by definition because they match what the
// target would have generated, so the target got them for the price of one
// forward pass. The first disagreement ends the run, and the target's own token
// at that slot becomes the bonus token the caller appends.
func acceptPrefix(draft, target []int) int {
	n := 0
	for n < len(draft) && n < len(target) && draft[n] == target[n] {
		n++
	}
	return n
}
