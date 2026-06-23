package llama

import (
	"strings"
	"testing"
)

func TestSamplerGreedyTakesArgmax(t *testing.T) {
	s := newSampler(Sampling{Temperature: 0})
	if !s.greedy() {
		t.Fatal("temperature 0 should be greedy")
	}
	got := s.sample([]float32{0.1, 0.9, 0.4, -2.0})
	if got != 1 {
		t.Errorf("greedy pick = %d, want 1", got)
	}
}

func TestSamplerSeededIsDeterministic(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.5, 0.2, -1.0, 3.0, 0.7}
	a := newSampler(Sampling{Temperature: 0.8, TopP: 0.95, TopK: 5, Seed: 42})
	b := newSampler(Sampling{Temperature: 0.8, TopP: 0.95, TopK: 5, Seed: 42})
	for i := range 50 {
		if a.sample(logits) != b.sample(logits) {
			t.Fatalf("same seed diverged at step %d", i)
		}
	}
}

func TestSamplerStaysInVocab(t *testing.T) {
	s := newSampler(Sampling{Temperature: 1.0, Seed: 7})
	logits := []float32{0.3, 0.3, 0.3, 0.3}
	for range 100 {
		got := s.sample(logits)
		if got < 0 || got >= len(logits) {
			t.Fatalf("sampled out-of-range token %d", got)
		}
	}
}

func TestSamplerTopKLimitsChoices(t *testing.T) {
	// With top-k 1 the sampler can only ever return the single best token, even
	// at high temperature.
	s := newSampler(Sampling{Temperature: 2.0, TopK: 1, Seed: 1})
	logits := []float32{0.1, 5.0, 0.2, 0.3}
	for range 100 {
		if got := s.sample(logits); got != 1 {
			t.Fatalf("top-k 1 returned %d, want 1", got)
		}
	}
}

func TestAcceptPrefix(t *testing.T) {
	cases := []struct {
		draft, target []int
		want          int
	}{
		{[]int{5, 8, 2}, []int{5, 8, 2}, 3}, // full accept
		{[]int{5, 8, 2}, []int{5, 8, 9}, 2}, // diverge at last
		{[]int{5, 8, 2}, []int{1, 8, 2}, 0}, // diverge immediately
		{[]int{5}, []int{5, 8, 2}, 1},       // draft shorter
		{[]int{5, 8, 2, 7}, []int{5, 8}, 2}, // target shorter
		{[]int{}, []int{5}, 0},              // empty draft
	}
	for i, c := range cases {
		if got := acceptPrefix(c.draft, c.target); got != c.want {
			t.Errorf("case %d: acceptPrefix = %d, want %d", i, got, c.want)
		}
	}
}

func TestStopFilterReleasesText(t *testing.T) {
	f := newStopFilter([]string{"STOP"})
	out, stopped := f.push("hello world")
	if stopped {
		t.Fatal("unexpected stop")
	}
	if out != "hello world" {
		t.Errorf("emit = %q, want full text", out)
	}
}

func TestStopFilterCutsAtSequence(t *testing.T) {
	f := newStopFilter([]string{"<end>"})
	var got strings.Builder
	for _, piece := range []string{"the answer is ", "42", "<end>", " ignored"} {
		out, stopped := f.push(piece)
		got.WriteString(out)
		if stopped {
			break
		}
	}
	if got.String() != "the answer is 42" {
		t.Errorf("emitted %q, want %q", got.String(), "the answer is 42")
	}
}

func TestStopFilterHandlesSplitSequence(t *testing.T) {
	// The stop sequence "<|im_end|>" arrives split across several pieces; nothing
	// before it should be held back forever, and the sequence must still trigger.
	f := newStopFilter([]string{"<|im_end|>"})
	var got strings.Builder
	pieces := []string{"done", "<|", "im_", "end", "|>", "more"}
	stoppedAt := -1
	for i, piece := range pieces {
		out, stopped := f.push(piece)
		got.WriteString(out)
		if stopped {
			stoppedAt = i
			break
		}
	}
	if got.String() != "done" {
		t.Errorf("emitted %q, want %q", got.String(), "done")
	}
	if stoppedAt != 4 {
		t.Errorf("stopped at piece %d, want 4", stoppedAt)
	}
}

func TestStopFilterPartialMatchThatFails(t *testing.T) {
	// A run that looks like the start of the stop sequence but then diverges must
	// be released in full, not swallowed.
	f := newStopFilter([]string{"STOP"})
	var got strings.Builder
	for _, piece := range []string{"ST", "AR", "T over"} {
		emit, stopped := f.push(piece)
		got.WriteString(emit)
		if stopped {
			t.Fatal("should not have stopped")
		}
	}
	got.WriteString(f.flush())
	if got.String() != "START over" {
		t.Errorf("emitted %q, want %q", got.String(), "START over")
	}
}

func TestStopFilterNoStops(t *testing.T) {
	f := newStopFilter(nil)
	out, stopped := f.push("anything")
	if stopped || out != "anything" {
		t.Errorf("no-stop filter changed output: %q stopped=%v", out, stopped)
	}
}
