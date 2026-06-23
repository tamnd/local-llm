package auth

import (
	"net/http"
	"testing"
)

func TestValid(t *testing.T) {
	a := New([]string{"sk-local-abc", "sk-local-def", ""})
	if !a.Valid("sk-local-abc") {
		t.Error("known token should be valid")
	}
	if !a.Valid("sk-local-def") {
		t.Error("second known token should be valid")
	}
	if a.Valid("sk-local-xyz") {
		t.Error("unknown token should be invalid")
	}
	if a.Valid("") {
		t.Error("empty token should be invalid")
	}
}

func TestPrefix(t *testing.T) {
	if got := Prefix("sk-local-abc123"); got != "sk-local" {
		t.Errorf("Prefix = %q, want sk-local", got)
	}
	if got := Prefix("short"); got != "short" {
		t.Errorf("Prefix of short token = %q, want short", got)
	}
}

func TestBearerToken(t *testing.T) {
	cases := []struct {
		header string
		want   string
		ok     bool
	}{
		{"Bearer sk-local-abc", "sk-local-abc", true},
		{"bearer sk-local-abc", "sk-local-abc", true}, // case-insensitive scheme
		{"Bearer  sk-local-abc ", "sk-local-abc", true},
		{"Basic abc", "", false},
		{"", "", false},
		{"Bearer", "", false},
	}
	for _, tc := range cases {
		got, ok := BearerToken(tc.header)
		if got != tc.want || ok != tc.ok {
			t.Errorf("BearerToken(%q) = %q,%v want %q,%v", tc.header, got, ok, tc.want, tc.ok)
		}
	}
}

func TestCheck(t *testing.T) {
	a := New([]string{"sk-local-abc"})

	r, _ := http.NewRequest("GET", "/v1/models", nil)
	if _, e := a.Check(r); e == nil || e.Code != "missing_auth" {
		t.Errorf("no header: want missing_auth, got %+v", e)
	}

	r.Header.Set("Authorization", "Bearer sk-wrong")
	if _, e := a.Check(r); e == nil || e.Code != "invalid_token" {
		t.Errorf("wrong token: want invalid_token, got %+v", e)
	}

	r.Header.Set("Authorization", "Bearer sk-local-abc")
	tok, e := a.Check(r)
	if e != nil || tok != "sk-local-abc" {
		t.Errorf("good token: want sk-local-abc,nil got %q,%+v", tok, e)
	}
}
