package llama

import "strings"

// stopFilter makes stop sequences work over a streamed token output. A stop
// string can straddle token boundaries, and once a piece is streamed to the
// client it cannot be taken back, so the filter holds back the smallest tail
// that could still turn out to be the start of a stop sequence and releases the
// rest. When a stop sequence completes, the text up to it is released and
// everything from the sequence onward is dropped.
//
// It is pure Go and unit-tested, which is where the awkward boundary cases live
// (a stop sequence split across three tokens, one stop that is a prefix of
// another, a partial match that turns out not to be one). The cgo decode loop
// feeds it each decoded piece and streams whatever it returns.
type stopFilter struct {
	stops  []string
	maxLen int
	buf    string // text accepted but not yet safe to emit
}

func newStopFilter(stops []string) *stopFilter {
	f := &stopFilter{}
	for _, s := range stops {
		if s == "" {
			continue
		}
		f.stops = append(f.stops, s)
		if len(s) > f.maxLen {
			f.maxLen = len(s)
		}
	}
	return f
}

// push adds the next decoded piece. It returns the text that is now safe to emit
// and whether a stop sequence has completed. When stopped is true the caller
// emits the returned text and ends the generation; the stop sequence itself and
// anything after it are not emitted.
func (f *stopFilter) push(piece string) (emit string, stopped bool) {
	if len(f.stops) == 0 {
		return piece, false
	}
	f.buf += piece

	// If a stop sequence is fully present, cut at its earliest occurrence.
	cut := -1
	for _, s := range f.stops {
		if i := strings.Index(f.buf, s); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut >= 0 {
		out := f.buf[:cut]
		f.buf = ""
		return out, true
	}

	// No complete match. Hold back the longest suffix of buf that is a prefix of
	// some stop sequence, since the next piece could complete it. Release the rest.
	hold := f.heldSuffixLen()
	out := f.buf[:len(f.buf)-hold]
	f.buf = f.buf[len(f.buf)-hold:]
	return out, false
}

// flush returns whatever is still held back when generation ends without hitting
// a stop sequence. The held tail was never part of a real stop, so it belongs in
// the output.
func (f *stopFilter) flush() string {
	out := f.buf
	f.buf = ""
	return out
}

// heldSuffixLen is the length of the longest suffix of buf that equals a prefix
// of any stop sequence (shorter than the whole sequence, since a full match was
// already ruled out). That suffix is the only part that could still grow into a
// stop, so it is the only part worth holding back.
func (f *stopFilter) heldSuffixLen() int {
	limit := min(f.maxLen, len(f.buf))
	for n := limit; n > 0; n-- {
		suffix := f.buf[len(f.buf)-n:]
		for _, s := range f.stops {
			if len(s) > n && strings.HasPrefix(s, suffix) {
				return n
			}
		}
	}
	return 0
}
