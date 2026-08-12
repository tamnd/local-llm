package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	// Registered for their DecodeConfig side effect: the bench labels each vision
	// row with the image's pixel dimensions, which is the variable that drives
	// how many tokens the projector emits.
	_ "image/jpeg"
	_ "image/png"
)

// contentPart is one element of an OpenAI multimodal message. A part is either
// text (Text set) or an image (ImageURL set), never both.
type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

// visionInput is one benchable image: the data URL the request carries plus the
// metadata the result row records.
type visionInput struct {
	path    string
	dataURL string
	width   int
	height  int
	bytes   int
}

// loadImage reads an image off disk and prepares it for inline transport. The
// gateway and llama-server both accept a base64 data URL, which keeps the bench
// self-contained: no static file server has to be reachable from the box, and
// the image bytes travel the same path a real client's would.
func loadImage(path string) (visionInput, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return visionInput{}, err
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return visionInput{}, fmt.Errorf("%s: decode header: %w", filepath.Base(path), err)
	}
	mime := "image/" + format
	return visionInput{
		path:    path,
		dataURL: "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw),
		width:   cfg.Width,
		height:  cfg.Height,
		bytes:   len(raw),
	}, nil
}

// visionContent builds the message body for a vision cell: an optional salt,
// then the images, then the instruction. Image-before-instruction is the order
// llama.cpp's mtmd path and the Muse Glimmer chat template expect; putting the
// instruction first makes some templates drop the image marker entirely and the
// cell then silently measures a text-only generation.
//
// The salt has to lead. A prompt cache matches on the longest common prefix, so
// a salt placed after the image would leave the image's tokens cached and the
// rep would still skip the encode it is there to measure (see saltPrompt). A
// few tokens ahead of it invalidate the whole thing.
func visionContent(salt, prompt string, imgs []visionInput) []contentPart {
	parts := make([]contentPart, 0, len(imgs)+2)
	if salt != "" {
		parts = append(parts, contentPart{Type: "text", Text: salt})
	}
	for _, im := range imgs {
		parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: im.dataURL}})
	}
	parts = append(parts, contentPart{Type: "text", Text: prompt})
	return parts
}

// visionLabel renders the pixel dimensions of a cell's images for the result
// row, e.g. "1024x1024" for one image or "1024x1024+512x512" for two.
func visionLabel(imgs []visionInput) string {
	var b strings.Builder
	for i, im := range imgs {
		if i > 0 {
			b.WriteByte('+')
		}
		fmt.Fprintf(&b, "%dx%d", im.width, im.height)
	}
	return b.String()
}
