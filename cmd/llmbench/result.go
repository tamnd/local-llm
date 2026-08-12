package main

import "encoding/json"

// result is one benched cell serialized to a JSON line. The field set and the
// json tags follow spec 2065 doc 11 section 2.7 so a run appends rows that a
// later cross-backend comparison reads without a schema migration. Speed lives
// here; correctness is out of scope for this file (see the solve runner).
type result struct {
	Date       string `json:"date"`
	Model      string `json:"model"`
	Quant      string `json:"quant"`
	Runtime    string `json:"runtime"`
	RuntimeVer string `json:"runtime_ver,omitempty"`
	Mode       string `json:"mode"`
	// Arm names the server configuration a row was measured under, so a file can
	// hold an A/B (say a runtime bump against a flag change) without the reader
	// having to date-match rows to a changelog. Blank on single-config runs.
	Arm          string  `json:"arm,omitempty"`
	PromptToks   int     `json:"prompt_toks"`
	GenToks      int     `json:"gen_toks"`
	Context      int     `json:"context"`
	GPU          string  `json:"gpu"`
	Driver       string  `json:"driver,omitempty"`
	VRAMUsedGB   float64 `json:"vram_used_gb"`
	MeanToksS    float64 `json:"mean_toks_s"`
	StdToksS     float64 `json:"std_toks_s"`
	PrefillToksS float64 `json:"prefill_toks_s"`
	TTFTms       int     `json:"ttft_ms"`
	Reps         int     `json:"reps"`

	// Vision-only fields, omitted on text rows so the file stays readable and a
	// mode=decode row is byte-identical to what earlier runs wrote. NImages and
	// ImagePx describe the input; ImageToks is the projector's contribution to
	// the prompt, derived by subtracting a text-only control from prompt_toks.
	NImages   int    `json:"n_images,omitempty"`
	ImagePx   string `json:"image_px,omitempty"`
	ImageToks int    `json:"image_toks,omitempty"`
}

func (r result) marshal() (string, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
