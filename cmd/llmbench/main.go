// Command llmbench is the cross-backend throughput bench. It drives any
// OpenAI-compatible endpoint (in practice the llmgw gateway, so one run covers
// ollama, vLLM, and TabbyAPI/ExLlamaV3 behind the same API) and records decode
// tok/s, prefill tok/s, and TTFT across a context-length sweep. Results land as
// JSON lines in bench/results/<date>.jsonl in the schema fixed by spec 2065
// doc 11 section 2.7, so a later run appends comparable rows rather than
// overwriting a single-backend file the way bench/ollama-bench.ps1 does.
//
// The tool measures speed only. Correctness on real tasks is a separate
// dimension the solve runner (scripts/solve-swebench.sh) covers by pointing the
// tomo-labs swebench-live harness at the same gateway.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tamnd/local-llm/config"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "llmbench:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("llmbench", flag.ContinueOnError)
	base := fs.String("base-url", "http://100.71.238.128:8888/v1", "OpenAI-compatible base URL (the gateway data plane)")
	token := fs.String("token", envOr("LLMBENCH_TOKEN", ""), "bearer token for the gateway (or set LLMBENCH_TOKEN)")
	models := fs.String("models", "", "comma-separated gateway model ids to bench (required)")
	cfgPath := fs.String("config", "configs/llmgw.yaml", "gateway config, read to label each model's runtime and quant")
	contexts := fs.String("contexts", "512,2048,8192,32768", "comma-separated prompt-token targets to sweep")
	images := fs.String("images", "", "comma-separated image files; each is benched as one mode=vision cell (requires a model served with an --mmproj projector)")
	visionPrompt := fs.String("vision-prompt", "Describe this image in one sentence.", "instruction sent alongside each image in a vision cell")
	gen := fs.Int("gen", 256, "tokens to generate per run")
	reps := fs.Int("reps", 3, "measured reps per (model,context) cell, after one warm-up")
	timeout := fs.Duration("timeout", 5*time.Minute, "per-request timeout")
	out := fs.String("out", "", "output JSONL path (default bench/results/<date>.jsonl)")
	gpu := fs.String("gpu", "RTX 4090", "GPU label recorded in each row")
	arm := fs.String("arm", "", "server-configuration label recorded in each row, e.g. \"b10380+mmproj\"; lets one results file hold an A/B")
	runtimeVer := fs.String("runtime-ver", "", "runtime version recorded in each row, overriding the config's runtime_ver")
	driver := fs.String("driver", "", "driver version recorded in each row")
	vramCmd := fs.String("vram-cmd", "", "shell command whose stdout is the used VRAM in MiB, sampled once per cell (e.g. an ssh nvidia-smi call); blank leaves vram_used_gb at 0")
	reuse := fs.Bool("reuse-prompt", false, "send byte-identical prompts across reps, as runs before 2026-08-12 did; leaves prefill_toks_s and ttft_ms inflated by llama-server's prompt cache, and exists only to reproduce those older rows")
	dryRun := fs.Bool("dry-run", false, "print the plan and exit without calling the endpoint")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*models) == "" {
		return fmt.Errorf("-models is required (comma-separated gateway model ids)")
	}

	modelIDs := splitTrim(*models)
	ctxTargets, err := parseInts(*contexts)
	if err != nil {
		return fmt.Errorf("-contexts: %w", err)
	}

	// Images are loaded and header-decoded up front so a bad path or an
	// unsupported format fails before the first request rather than midway
	// through a sweep that has already appended rows.
	var visionInputs []visionInput
	for _, p := range splitTrim(*images) {
		im, err := loadImage(p)
		if err != nil {
			return fmt.Errorf("-images: %w", err)
		}
		visionInputs = append(visionInputs, im)
	}

	// The config is a labelling aid, not a hard dependency: it lets a row say
	// runtime=tabby/quant=exl3 without a per-model flag. A model missing from the
	// config still benches; it is just labelled runtime=unknown.
	labels := map[string]modelLabel{}
	if cfg, err := config.Load(*cfgPath); err == nil {
		for id, m := range cfg.Models {
			labels[id] = labelFor(m)
		}
	} else {
		fmt.Fprintf(os.Stderr, "llmbench: config %s not read (%v); rows will be labelled runtime=unknown\n", *cfgPath, err)
	}

	date := time.Now().Format("20060102")
	outPath := *out
	if outPath == "" {
		outPath = filepath.Join("bench", "results", date+".jsonl")
	}

	plan := fmt.Sprintf("bench %d model(s) x (%d context(s) + %d image(s)) x %d rep(s) -> %s",
		len(modelIDs), len(ctxTargets), len(visionInputs), *reps, outPath)
	fmt.Fprintln(os.Stderr, plan)
	if *dryRun {
		for _, id := range modelIDs {
			l := labels[id]
			fmt.Fprintf(os.Stderr, "  %s  runtime=%s quant=%s  contexts=%v gen=%d\n", id, orUnknown(l.runtime), orUnknown(l.quant), ctxTargets, *gen)
			for _, im := range visionInputs {
				fmt.Fprintf(os.Stderr, "    vision %s (%dx%d, %d KiB)\n", filepath.Base(im.path), im.width, im.height, im.bytes/1024)
			}
		}
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	client := &http.Client{Timeout: *timeout}
	now := time.Now

	for _, id := range modelIDs {
		l := labels[id]
		for _, target := range ctxTargets {
			prompt := fillerPrompt(target)
			row, err := benchCell(client, *base, *token, id, prompt, *gen, *reps, *timeout, *reuse, now)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  SKIP %s @ ctx~%d: %v\n", id, target, err)
				continue
			}
			row.Date = date
			row.Model = id
			row.Runtime = orUnknown(l.runtime)
			row.Quant = orUnknown(l.quant)
			row.RuntimeVer = firstNonEmpty(*runtimeVer, l.runtimeVer)
			row.Arm = *arm
			row.Context = target
			row.GPU = *gpu
			row.Driver = *driver
			row.VRAMUsedGB = sampleVRAM(*vramCmd)
			line, err := row.marshal()
			if err != nil {
				return err
			}
			if _, err := f.WriteString(line + "\n"); err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "  %s @ ctx=%d (prompt=%d gen=%d): decode %.1f tok/s (sd %.1f), prefill %.0f tok/s, ttft %dms\n",
				id, target, row.PromptToks, row.GenToks, row.MeanToksS, row.StdToksS, row.PrefillToksS, row.TTFTms)
		}

		// The vision cells share the model loop so one invocation produces a
		// directly comparable text and image row set for the same server.
		for _, im := range visionInputs {
			row, err := benchVisionCell(client, *base, *token, id, *visionPrompt, im, *gen, *reps, *timeout, *reuse, now)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  SKIP %s @ vision %s: %v\n", id, filepath.Base(im.path), err)
				continue
			}
			row.Date = date
			row.Model = id
			row.Runtime = orUnknown(l.runtime)
			row.Quant = orUnknown(l.quant)
			row.RuntimeVer = firstNonEmpty(*runtimeVer, l.runtimeVer)
			row.Arm = *arm
			row.GPU = *gpu
			row.Driver = *driver
			row.VRAMUsedGB = sampleVRAM(*vramCmd)
			line, err := row.marshal()
			if err != nil {
				return err
			}
			if _, err := f.WriteString(line + "\n"); err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "  %s @ vision %s (%s, prompt=%d of which ~%d image, gen=%d): decode %.1f tok/s (sd %.1f), prefill %.0f tok/s, ttft %dms\n",
				id, filepath.Base(im.path), row.ImagePx, row.PromptToks, row.ImageToks, row.GenToks, row.MeanToksS, row.StdToksS, row.PrefillToksS, row.TTFTms)
		}
	}
	fmt.Fprintln(os.Stderr, "done ->", outPath)
	return nil
}

// benchCell runs one warm-up plus reps measured generations for a single
// (model, context) pair and folds them into one result row. The warm-up absorbs
// the cold-load tax (weights paging into VRAM, doc 11) so it never impersonates
// a slow decode; only the measured reps feed the mean and standard deviation.
func benchCell(client *http.Client, base, token, model, prompt string, gen, reps int, timeout time.Duration, reuse bool, now func() time.Time) (result, error) {
	warmCtx, cancel := context.WithTimeout(context.Background(), timeout)
	// The warm-up only absorbs the cold-load tax, so its token count is
	// throwaway, but it must not be 1: a thinking model (qwen3) spends its first
	// token on the <think> open marker, which streams as an empty delta (no
	// content, no reasoning). A 1-token budget then sees nothing generated and
	// the whole cell is wrongly skipped, so give it enough headroom to surface a
	// real token before the length cutoff.
	_, err := measure(warmCtx, client, base, token, model, "warm up: reply with one word.", 16, now)
	cancel()
	if err != nil {
		return result{}, fmt.Errorf("warm-up: %w", err)
	}

	var decodeRates []float64
	var last sample
	var prefillSum float64
	var ttftSum time.Duration
	for i := 0; i < reps; i++ {
		rc, cancel := context.WithTimeout(context.Background(), timeout)
		s, err := measure(rc, client, base, token, model, saltPrompt(prompt, i, reuse), gen, now)
		cancel()
		if err != nil {
			return result{}, fmt.Errorf("rep %d: %w", i+1, err)
		}
		decodeRates = append(decodeRates, s.decodeToksPerSec())
		prefillSum += s.prefillToksPerSec()
		ttftSum += s.ttft
		last = s
	}
	mean, std := meanStd(decodeRates)
	return result{
		Mode:         "decode",
		PromptToks:   last.promptToks,
		GenToks:      last.genToks,
		MeanToksS:    round1(mean),
		StdToksS:     round1(std),
		PrefillToksS: round1(prefillSum / float64(reps)),
		TTFTms:       int((ttftSum / time.Duration(reps)).Milliseconds()),
		Reps:         reps,
	}, nil
}

// benchVisionCell is benchCell's multimodal twin: same warm-up, same rep count,
// same folding into one row, but the message carries an image part so TTFT
// includes the vision encode and the projector's tokens land in prompt_toks.
//
// It also runs one text-only control with the identical instruction. The
// difference in prompt_tokens is what the image itself cost, which is the
// number that explains a vision TTFT and which no backend reports directly. The
// control is a single unmeasured request, so it costs one short generation.
func benchVisionCell(client *http.Client, base, token, model, prompt string, im visionInput, gen, reps int, timeout time.Duration, reuse bool, now func() time.Time) (result, error) {
	// The control is the measured message with the image part removed and
	// nothing else changed: same salt, same parts, same order. Anything less
	// leaves salt or part-boundary tokens in the difference and image_toks comes
	// out a few tokens wide.
	ctrlCtx, cancel := context.WithTimeout(context.Background(), timeout)
	ctrl, err := measureContent(ctrlCtx, client, base, token, model, visionContent(saltFor(reps-1, reuse), prompt, nil), 16, now)
	cancel()
	if err != nil {
		return result{}, fmt.Errorf("text control: %w", err)
	}

	warmCtx, cancel := context.WithTimeout(context.Background(), timeout)
	_, err = measureContent(warmCtx, client, base, token, model, visionContent("", prompt, []visionInput{im}), 16, now)
	cancel()
	if err != nil {
		return result{}, fmt.Errorf("warm-up: %w", err)
	}

	var decodeRates []float64
	var last sample
	var prefillSum float64
	var ttftSum time.Duration
	for i := 0; i < reps; i++ {
		rc, cancel := context.WithTimeout(context.Background(), timeout)
		content := visionContent(saltFor(i, reuse), prompt, []visionInput{im})
		s, err := measureContent(rc, client, base, token, model, content, gen, now)
		cancel()
		if err != nil {
			return result{}, fmt.Errorf("rep %d: %w", i+1, err)
		}
		decodeRates = append(decodeRates, s.decodeToksPerSec())
		prefillSum += s.prefillToksPerSec()
		ttftSum += s.ttft
		last = s
	}
	mean, std := meanStd(decodeRates)

	// A model served without --mmproj usually still answers: the template drops
	// the image and it describes nothing. Equal prompt token counts are the
	// tell, and reporting that as a vision result would be a lie, so fail loudly.
	imageToks := last.promptToks - ctrl.promptToks
	if imageToks <= 0 {
		return result{}, fmt.Errorf("image added %d prompt tokens (text control %d, with image %d): the server is ignoring the image, check --mmproj",
			imageToks, ctrl.promptToks, last.promptToks)
	}

	return result{
		Mode:         "vision",
		PromptToks:   last.promptToks,
		GenToks:      last.genToks,
		Context:      last.promptToks,
		NImages:      1,
		ImagePx:      visionLabel([]visionInput{im}),
		ImageToks:    imageToks,
		MeanToksS:    round1(mean),
		StdToksS:     round1(std),
		PrefillToksS: round1(prefillSum / float64(reps)),
		TTFTms:       int((ttftSum / time.Duration(reps)).Milliseconds()),
		Reps:         reps,
	}, nil
}

// modelLabel is the descriptive metadata a row inherits from the gateway config
// so the bench does not have to be told each model's runtime by hand.
type modelLabel struct {
	runtime    string
	quant      string
	runtimeVer string
}

// labelFor derives the runtime and a best-effort quant label from a config
// model entry. The quant is read from a params.quant hint if present, else
// guessed from the upstream model name (fp8, awq, exl3, gguf), so the row is
// self-describing without a new required config field.
func labelFor(m config.ModelEntry) modelLabel {
	l := modelLabel{runtime: m.Backend}
	if q, ok := m.Params["quant"].(string); ok && q != "" {
		l.quant = q
	} else {
		l.quant = guessQuant(m.Backend, m.UpstreamModel)
	}
	if v, ok := m.Params["runtime_ver"].(string); ok {
		l.runtimeVer = v
	}
	return l
}

func guessQuant(backend, upstream string) string {
	u := strings.ToLower(upstream)
	switch {
	case strings.Contains(u, "fp8"):
		return "fp8"
	case strings.Contains(u, "awq"):
		return "awq"
	case strings.Contains(u, "exl3"):
		return "exl3"
	case strings.Contains(u, "exl2"):
		return "exl2"
	case strings.Contains(u, "bpw"):
		return "exl"
	}
	switch backend {
	case config.BackendOllama:
		return "gguf"
	case config.BackendInproc, config.BackendLlama:
		return "gguf"
	}
	return ""
}

// sampleVRAM runs the optional VRAM probe command and parses the first integer
// (MiB used) from its stdout, converting to GiB. A blank command or a parse
// miss records 0 rather than failing the whole run, because the bench measures
// from the Mac and the GPU lives on the box, so VRAM is an out-of-band sample.
func sampleVRAM(cmdStr string) float64 {
	if strings.TrimSpace(cmdStr) == "" {
		return 0
	}
	out, err := exec.Command("sh", "-c", cmdStr).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "  vram probe failed: %v\n", err)
		return 0
	}
	fields := strings.FieldsFunc(string(out), func(r rune) bool { return r < '0' || r > '9' })
	if len(fields) == 0 {
		return 0
	}
	mib, err := strconv.Atoi(fields[0])
	if err != nil {
		return 0
	}
	return round1(float64(mib) / 1024.0)
}

// saltPrompt gives rep i its own prompt prefix so llama-server's prompt cache
// cannot serve it from the previous rep.
//
// This matters more than it looks. Without a salt, every rep in a cell sends
// byte-identical text: rep 1 pays the real prefill, reps 2..n hit the cache and
// return a first token in a fraction of the time. The mean TTFT then sits well
// below the true one and prefill_toks_s, which is derived from it, reads high by
// an order of magnitude at long contexts. Measured on this box at 32K: 8.6 s
// uncached against 0.7 s cached, i.e. 2.9k tok/s of real prefill reported as
// 26-38k. Rows written before this fix (bench/results/20260810.jsonl and
// earlier) carry that inflation in prefill_toks_s and ttft_ms; their decode
// figures are unaffected, because the cache only shortcuts prefill.
//
// reuse restores the old behaviour for a like-for-like rerun against those rows.
func saltPrompt(prompt string, rep int, reuse bool) string {
	if reuse {
		return prompt
	}
	return saltFor(rep, reuse) + " " + prompt
}

// saltFor is the salt text alone, for callers that place it in its own message
// part rather than splicing it into a prompt string.
func saltFor(rep int, reuse bool) string {
	if reuse {
		return ""
	}
	return fmt.Sprintf("[rep %d]", rep)
}

// fillerPrompt builds a prompt whose token count approximates target. One
// English word is roughly 1.3 tokens, so target/1.3 words lands close; the row
// records the server's exact prompt_tokens, so the approximation only sets the
// sweep point, never the reported number.
func fillerPrompt(target int) string {
	if target <= 8 {
		return "Summarize the number one."
	}
	words := int(float64(target) / 1.3)
	var b strings.Builder
	b.WriteString("Read the following log and reply with a one sentence summary.\n\n")
	filler := []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "while", "the", "server", "logs", "each", "request", "with", "a", "latency", "and", "status", "code"}
	for i := 0; i < words; i++ {
		b.WriteString(filler[i%len(filler)])
		b.WriteByte(' ')
	}
	return b.String()
}

func meanStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(len(xs))
	if len(xs) < 2 {
		return mean, 0
	}
	var sq float64
	for _, x := range xs {
		d := x - mean
		sq += d * d
	}
	return mean, math.Sqrt(sq / float64(len(xs)-1))
}

func round1(x float64) float64 { return math.Round(x*10) / 10 }

func splitTrim(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, p := range splitTrim(s) {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	sort.Ints(out)
	return out, nil
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}
