package observe

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RotatingWriter is an io.WriteCloser that appends to a log file and rolls it
// over once it crosses a size threshold, keeping a bounded number of old files.
// It is deliberately small: the gateway writes one short JSON line per request,
// so a size-based roll with a handful of kept files is all the stack needs. No
// external logging dependency is pulled in for this.
//
// Rotation renames log -> log.1 -> log.2 ... up to keep files, dropping the
// oldest. This is the same scheme doc 12 section 2 specifies (rotate at 10-100
// MB, keep 5).
type RotatingWriter struct {
	mu       sync.Mutex
	path     string
	maxBytes int64
	keep     int
	f        *os.File
	size     int64
}

// NewFile opens (creating if needed) the log at path. rotateMB is the size
// threshold in mebibytes; keep is how many rolled files to retain. A rotateMB
// of zero disables rotation (the file grows unbounded), which is only sensible
// in tests.
func NewFile(path string, rotateMB, keep int) (*RotatingWriter, error) {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("observe: create log dir %s: %w", dir, err)
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("observe: open log %s: %w", path, err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("observe: stat log %s: %w", path, err)
	}
	return &RotatingWriter{
		path:     path,
		maxBytes: int64(rotateMB) * 1 << 20,
		keep:     keep,
		f:        f,
		size:     info.Size(),
	}, nil
}

// Write appends p, rotating first if the file would cross the threshold. It
// satisfies io.Writer.
func (w *RotatingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.maxBytes > 0 && w.size+int64(len(p)) > w.maxBytes {
		if err := w.rotate(); err != nil {
			return 0, err
		}
	}
	n, err := w.f.Write(p)
	w.size += int64(n)
	return n, err
}

// Close closes the underlying file.
func (w *RotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	return err
}

// rotate closes the current file, shifts the numbered backups, and opens a
// fresh primary file. The caller holds the lock.
func (w *RotatingWriter) rotate() error {
	if err := w.f.Close(); err != nil {
		return fmt.Errorf("observe: close before rotate: %w", err)
	}
	// Drop the oldest, then shift each backup up by one: .(keep-1) -> .keep,
	// ... , .1 -> .2, primary -> .1.
	oldest := fmt.Sprintf("%s.%d", w.path, w.keep)
	_ = os.Remove(oldest) // best effort; missing file is fine.
	for i := w.keep - 1; i >= 1; i-- {
		src := fmt.Sprintf("%s.%d", w.path, i)
		dst := fmt.Sprintf("%s.%d", w.path, i+1)
		_ = os.Rename(src, dst) // best effort; gaps are tolerated.
	}
	if w.keep > 0 {
		_ = os.Rename(w.path, w.path+".1")
	}
	f, err := os.OpenFile(w.path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("observe: reopen after rotate: %w", err)
	}
	w.f = f
	w.size = 0
	return nil
}
