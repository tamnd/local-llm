// Command llmgw is the local-LLM gateway: an OpenAI-compatible reverse proxy
// that fronts Ollama, llama-server, TabbyAPI, and vLLM on a single RTX 4090 box,
// routing by model name and swapping models in and out of VRAM so one large
// model fits the 24 GB card. It binds two listeners: the data plane (the API
// surface, reachable over the tailnet) and a loopback-only admin plane.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/tamnd/local-llm/backend"
	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/gateway"
	"github.com/tamnd/local-llm/manager"
	"github.com/tamnd/local-llm/observe"
	"github.com/tamnd/local-llm/router"
)

// version is stamped by the linker at release time
// (-ldflags "-X main.version=v0.1.0"). It feeds the system_fingerprint and the
// /healthz response.
var version = "dev"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "llmgw:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("llmgw", flag.ContinueOnError)
	configPath := fs.String("config", "configs/llmgw.yaml", "path to the gateway config file")
	showVersion := fs.Bool("version", false, "print version and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *showVersion {
		fmt.Println("llmgw", version)
		return nil
	}

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	log, closer, err := observe.FromConfig(cfg.Logging, os.Stdout)
	if err != nil {
		return fmt.Errorf("init logging: %w", err)
	}
	defer func() { _ = closer.Close() }()

	reg := backend.NewRegistry(version)
	mgr := manager.New(cfg, reg, log)
	rt := router.New(cfg)
	gw := gateway.New(cfg, rt, mgr, reg, log, version)
	handler := gw.Handler()

	// The data plane and admin plane bind separately so the admin plane can stay
	// on loopback while the data plane is reachable over the tailnet. Both serve
	// the same mux; the gateway enforces which routes each token may reach.
	apiSrv := &http.Server{
		Addr:              cfg.Bind.APIAddr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}
	adminSrv := &http.Server{
		Addr:              cfg.Bind.AdminAddr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}

	errc := make(chan error, 2)
	go serve(apiSrv, "api", log, errc)
	go serve(adminSrv, "admin", log, errc)

	log.Info("gateway_started", map[string]any{
		"version":       version,
		"api_addr":      cfg.Bind.APIAddr,
		"admin_addr":    cfg.Bind.AdminAddr,
		"default_model": rt.Default(),
		"models":        len(rt.IDs()),
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		log.Info("gateway_stopping", nil)
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	apiErr := apiSrv.Shutdown(shutdownCtx)
	adminErr := adminSrv.Shutdown(shutdownCtx)
	return errors.Join(apiErr, adminErr)
}

// serve runs one HTTP server and reports a non-graceful exit on errc. A clean
// shutdown returns http.ErrServerClosed, which is not an error here.
func serve(srv *http.Server, name string, log *observe.Logger, errc chan<- error) {
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("listener_failed", map[string]any{"listener": name, "addr": srv.Addr, "error": err.Error()})
		errc <- fmt.Errorf("%s listener: %w", name, err)
	}
}
