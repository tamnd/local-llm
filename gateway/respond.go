package gateway

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/tamnd/local-llm/auth"
)

// writeJSON encodes v as the response body with the given status. Encoding
// errors are swallowed: the header is already committed and there is nothing
// useful to send a client that cannot receive our JSON.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// errorBody mirrors OpenAI's error envelope so existing clients surface the
// message unchanged.
type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// writeError sends an OpenAI-shaped error with an HTTP status. code is the
// machine-readable token clients can switch on; message is human-facing.
func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, errorBody{Error: errorDetail{Message: message, Type: "invalid_request_error", Code: code}})
}

// writeAuthError maps an auth failure to its HTTP status. A missing credential
// is 401 with a WWW-Authenticate challenge; a present-but-wrong one is 403.
func writeAuthError(w http.ResponseWriter, e *auth.AuthError) {
	status := http.StatusForbidden
	if e.Code == "missing_auth" || e.Code == "admin_disabled" {
		status = http.StatusUnauthorized
		w.Header().Set("WWW-Authenticate", "Bearer")
	}
	writeError(w, status, e.Code, e.Message)
}

// decodeJSON reads and decodes a JSON request body with the same size cap as the
// inference path, rejecting unknown fields so typos in admin calls are caught.
func decodeJSON(r *http.Request, v any) error {
	dec := json.NewDecoder(io.LimitReader(r.Body, maxBodyBytes))
	dec.DisallowUnknownFields()
	return dec.Decode(v)
}
