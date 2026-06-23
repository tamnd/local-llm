// Package auth implements the bearer-token check that guards the gateway's
// public API. It is the second security boundary behind the tailnet (doc 09):
// even a device on the Tailscale network must present a token from the config's
// auth.tokens list. The check is constant-time so a timing side channel cannot
// be used to recover a token byte by byte.
package auth

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// Authenticator holds the accepted tokens and answers two questions: is a
// presented token valid, and what is its log-safe prefix. Tokens are compared in
// constant time.
type Authenticator struct {
	tokens [][]byte
}

// New builds an Authenticator from the configured token list. Empty tokens are
// skipped; config validation already rejects them, this is belt and suspenders.
func New(tokens []string) *Authenticator {
	a := &Authenticator{}
	for _, t := range tokens {
		if t == "" {
			continue
		}
		a.tokens = append(a.tokens, []byte(t))
	}
	return a
}

// Valid reports whether the raw token matches one of the accepted tokens. The
// comparison runs against every configured token so the time taken does not
// reveal which token (if any) matched.
func (a *Authenticator) Valid(token string) bool {
	if token == "" {
		return false
	}
	got := []byte(token)
	ok := false
	for _, want := range a.tokens {
		if subtle.ConstantTimeCompare(got, want) == 1 {
			ok = true
		}
	}
	return ok
}

// Prefix returns the first 8 characters of a token for the request log. The
// full secret is never logged; doc 08 section 11 logs only this prefix so a
// device can be traced without leaking its credential.
func Prefix(token string) string {
	if len(token) <= 8 {
		return token
	}
	return token[:8]
}

// BearerToken extracts the token from an Authorization header value of the form
// "Bearer <token>". It returns the token and whether the header was well-formed.
func BearerToken(header string) (string, bool) {
	const prefix = "Bearer "
	if len(header) <= len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return "", false
	}
	return strings.TrimSpace(header[len(prefix):]), true
}

// AuthError describes why a request was rejected, mapping to the OpenAI-style
// error body the gateway returns (doc 08 section 8.1).
type AuthError struct {
	Code    string // missing_auth or invalid_token
	Message string
}

// Check validates the Authorization header on a request. It returns the
// authenticated token on success, or a non-nil *AuthError on failure. The
// gateway turns the error into a 401 JSON body.
func (a *Authenticator) Check(r *http.Request) (string, *AuthError) {
	header := r.Header.Get("Authorization")
	if header == "" {
		return "", &AuthError{Code: "missing_auth", Message: "Missing Authorization header"}
	}
	token, ok := BearerToken(header)
	if !ok {
		return "", &AuthError{Code: "missing_auth", Message: "Authorization header must be 'Bearer <token>'"}
	}
	if !a.Valid(token) {
		return "", &AuthError{Code: "invalid_token", Message: "Invalid bearer token"}
	}
	return token, nil
}
