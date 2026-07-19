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
	gen := fs.Int("gen", 256, "tokens to generate per run")
	reps := fs.Int("reps", 3, "measured reps per (model,context) cell, after one warm-up")
	timeout := fs.Duration("timeout", 5*time.Minute, "per-request timeout")
	out := fs.String("out", "", "output JSONL path (default bench/results/<date>.jsonl)")
	gpu := fs.String("gpu", "RTX 4090", "GPU label recorded in each row")
	driver := fs.String("driver", "", "driver version recorded in each row")
	vramCmd := fs.String("vram-cmd", "", "shell command whose stdout is the used VRAM in MiB, sampled once per cell (e.g. an ssh nvidia-smi call); blank leaves vram_used_gb at 0")
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

	plan := fmt.Sprintf("bench %d model(s) x %d context(s) x %d rep(s) -> %s", len(modelIDs), len(ctxTargets), *reps, outPath)
	fmt.Fprintln(os.Stderr, plan)
	if *dryRun {
		for _, id := range modelIDs {
			l := labels[id]
			fmt.Fprintf(os.Stderr, "  %s  runtime=%s quant=%s  contexts=%v gen=%d\n", id, orUnknown(l.runtime), orUnknown(l.quant), ctxTargets, *gen)
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
			row, err := benchCell(client, *base, *token, id, prompt, *gen, *reps, *timeout, now)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  SKIP %s @ ctx~%d: %v\n", id, target, err)
				continue
			}
			row.Date = date
			row.Model = id
			row.Runtime = orUnknown(l.runtime)
			row.Quant = orUnknown(l.quant)
			row.RuntimeVer = l.runtimeVer
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
	}
	fmt.Fprintln(os.Stderr, "done ->", outPath)
	return nil
}

// benchCell runs one warm-up plus reps measured generations for a single
// (model, context) pair and folds them into one result row. The warm-up absorbs
// the cold-load tax (weights paging into VRAM, doc 11) so it never impersonates
// a slow decode; only the measured reps feed the mean and standard deviation.
func benchCell(client *http.Client, base, token, model, prompt string, gen, reps int, timeout time.Duration, now func() time.Time) (result, error) {
	warmCtx, cancel := context.WithTimeout(context.Background(), timeout)
	_, err := measure(warmCtx, client, base, token, model, "warm up: reply with one word.", 1, now)
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
		s, err := measure(rc, client, base, token, model, prompt, gen, now)
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

func orUnknown(s string) string {
	if s == "" {
		return "unknown"
	}
	return s
}
