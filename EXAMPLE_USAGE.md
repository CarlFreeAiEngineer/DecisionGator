# DecisionGator examples

DecisionGator is a software component: include it, call a function, use the answer. These examples show the same three calls in every supported language. To run a complete program in your language right away, see [examples/](examples/README.md). Install and packaging details live in each language's own guide: [Python](released/python/README.md), [Java](java/README.md), [Node.js](javascript/README.md), [browser](web/README.md), [C](code/README.md), [Rust](examples/rust/), and [Go](examples/go/).

Every language offers the same three ideas:

| Call                  | Returns                         | Use it when                                         |
| --------------------- | ------------------------------- | --------------------------------------------------- |
| `is_yes` / `isYes`    | boolean                         | You want a decision. Yes when probability >= 0.5.   |
| `is_yes_p` / `isYesP` | probability of yes, 0.0 to 1.0  | You want to pick your own cutoff or a review range. |
| `choose` / `chooseP`  | index of the best option        | You have several options, not just yes or no.       |

Two optional inputs apply to every call. **Criteria** are a pair of sentences spelling out what counts as yes and what counts as no. A **threshold** replaces the 0.5 cutoff; equality counts as yes. Errors are reported through the language's normal error mechanism and are never disguised as "no".

The first call loads the component automatically; later calls reuse it. Nothing leaves the process: no network, no API key, no server.

The model is experimental. See [measured accuracy](reports/accuracy-v4.md) before relying on any of these decisions.

## Python

```python
from decisiongator import is_yes, is_yes_p, choose

# A yes/no decision.
if is_yes("Any chance I could come in next Tuesday?",
          "Is this person asking for an appointment?"):
    offer_available_times()

# The probability, so you can keep an uncertain range for a human.
p = is_yes_p("Please return my money. The item arrived broken.",
             "Is the customer asking for a refund?")
if p >= 0.9:
    start_refund()
elif p >= 0.4:
    queue_for_review()

# Criteria and a stricter threshold.
cancel = is_yes(
    "Please cancel my subscription before the next renewal.",
    "Is the customer asking to cancel their subscription?",
    criteria={
        "yes": "The customer wants their subscription to end.",
        "no": "The customer asks about anything other than ending the subscription.",
    },
    threshold=0.9,
)

# Several options instead of yes/no. Returns the index of the best option.
team = choose(
    "My card was charged twice for last month's invoice.",
    "Which team should handle this message?",
    ["billing", "technical support", "sales"],
)   # 0, meaning "billing"
```

Failures raise `DecisionGatorError`, never return `False` or `None`; `choose` returns `None` only when the best option is below a threshold you passed. Invalid thresholds raise `ValueError`.

## Java

```java
import java.util.List;
import org.decisiongator.Criteria;
import org.decisiongator.Decisions;

// A yes/no decision.
if (Decisions.isYes(
        "Any chance I could come in next Tuesday?",
        "Is this person asking for an appointment?")) {
    offerAvailableTimes();
}

// The probability.
double p = Decisions.isYesP(
    "Please return my money. The item arrived broken.",
    "Is the customer asking for a refund?");

// Criteria and a stricter threshold.
var criteria = new Criteria(
    "The customer wants their subscription to end.",
    "The customer asks about anything other than ending the subscription.");
boolean cancel = Decisions.isYes(
    "Please cancel my subscription before the next renewal.",
    "Is the customer asking to cancel their subscription?",
    criteria, 0.90);

// Several options. Returns the index of the best option.
int team = Decisions.choose(
    "My card was charged twice for last month's invoice.",
    "Which team should handle this message?",
    List.of("billing", "technical support", "sales"));   // 0, meaning "billing"
```

Failures throw `DecisionGatorException`; invalid thresholds throw `IllegalArgumentException`. `Decisions.chooseP` returns the full ranking as a list of `Choice` values with `index()` and `p()`.

## TypeScript and JavaScript on Node.js

```typescript
import { isYes, isYesP, choose } from "decisiongator";

// A yes/no decision. Always await; a Promise itself is truthy.
if (await isYes(
    "The checkout page is failing for everyone on our team.",
    "Is this reporting a service outage?")) {
  alertSupport();
}

// The probability.
const p = await isYesP(
  "Please return my money. The item arrived broken.",
  "Is the customer asking for a refund?");

// Criteria and a stricter threshold.
const cancel = await isYes(
  "Please cancel my subscription before the next renewal.",
  "Is the customer asking to cancel their subscription?",
  {
    criteria: {
      yes: "The customer wants their subscription to end.",
      no: "The customer asks about anything other than ending the subscription.",
    },
    threshold: 0.9,
  });

// Several options. Resolves to the index of the best option.
const team = await choose(
  "My card was charged twice for last month's invoice.",
  "Which team should handle this message?",
  ["billing", "technical support", "sales"]);   // 0, meaning "billing"
```

Failures reject with `DecisionGatorError`, which carries a `code`. `chooseP` resolves to the full ranking as `{ index, p }` objects, best first.

## In a web browser

The same functions run inside the browser, on the user's device, from a WebAssembly build. The text never leaves the page.

```javascript
import { isYes, isYesP } from "decisiongator/web";

const needsReply = await isYes(
  "Could you let me know when my order will arrive?",
  "Is the customer asking for a reply?");
if (needsReply) {
  showReplyButton();
}

// Options are the same object as on Node.js.
const p = await isYesP(content, question, { criteria: { yes, no } });
```

Your web app bundles the WebAssembly component and its assets. Offline use requires those assets to be installed or cached first. Deployment, content-security-policy, and bundler notes are in [the browser guide](web/README.md).

## C# and .NET

```csharp
using DecisionGator;

// A yes/no decision.
if (Decisions.IsYes("Any chance I could come in next Tuesday?",
                    "Is this person asking for an appointment?"))
    OfferAvailableTimes();

// The probability.
double p = Decisions.IsYesP("Please return my money. The item arrived broken.",
                            "Is the customer asking for a refund?");

// Criteria and a stricter threshold.
bool cancel = Decisions.IsYes(
    "Please cancel my subscription before the next renewal.",
    "Is the customer asking to cancel their subscription?",
    new Criteria("The customer wants their subscription to end.",
                 "The customer asks about anything other than ending the subscription."),
    threshold: 0.9);

// Several options. Returns the index of the best option.
int team = Decisions.Choose(
    "My card was charged twice for last month's invoice.",
    "Which team should handle this message?",
    new[] { "billing", "technical support", "sales" });   // 0, meaning "billing"
```

Failures throw `DecisionGatorException` with a `StatusCode`; invalid thresholds throw `ArgumentOutOfRangeException`. `Decisions.ChooseP` returns every option as a `Choice(Index, P)` ranked best first. Point the wrapper at a bundle with `Bundle.Directory` or `DECISIONGATOR_BUNDLE`, or ship the bundle in a `decisiongator` folder beside the executable; see [the C# guide](csharp/README.md).

## C

```c
#include <stdio.h>
#include <string.h>
#include "decisiongator.h"

int main(void) {
    const char *content =
        "Report A: The app crashes when I export a report as PDF.\n"
        "Report B: Saving a report to PDF makes the application close.";
    const char *question = "Do these reports describe the same problem?";
    uint8_t yes;

    /* A yes/no decision. NULL means no criteria. */
    dg_status status = dg_is_yes(
        content, strlen(content), question, strlen(question), NULL, &yes);
    if (status != DG_OK) {
        fputs("Could not evaluate the question.\n", stderr);
        return 1;
    }
    puts(yes ? "Likely duplicate" : "Keep separate for review");

    /* The probability. */
    double p;
    if (dg_is_yes_p(content, strlen(content), question, strlen(question),
                    NULL, &p) == DG_OK) {
        printf("p_yes = %.3f\n", p);
    }

    /* Criteria and a stricter threshold. */
    const char *yes_text = "Both reports describe one underlying fault.";
    const char *no_text = "The reports describe different faults.";
    dg_criteria criteria = {
        yes_text, strlen(yes_text), no_text, strlen(no_text)};
    if (dg_is_yes_at_threshold(content, strlen(content),
                               question, strlen(question),
                               &criteria, 0.90, &yes) == DG_OK && yes) {
        puts("Confident duplicate");
    }
    return 0;
}
```

The status code is separate from the answer: on any error the output variable is left untouched. Call `dg_last_error` for a readable message. `dg_choose` and `dg_choose_p` rank several options; `dg_load`, `dg_evaluate`, and `dg_release` give explicit control of sessions and bundle paths. See [the header](code/include/decisiongator.h) and [the interface specification](specs/component-api.md).

## Rust

The component is itself a Rust library that exposes a C interface, so Rust calls it through `extern "C"` declarations. A working project with its `build.rs` is in [`examples/rust`](examples/rust); run it with `cargo run --release` from that directory. It links the bundle under `released/<platform>/` by default, or the directory named by `DECISIONGATOR_BUNDLE`.

```rust
use std::ffi::{c_char, c_void};

// Declarations matching decisiongator.h.
#[repr(C)]
struct DgCriteria {
    yes: *const c_char, yes_bytes: usize,
    no: *const c_char, no_bytes: usize,
}
extern "C" {
    fn dg_is_yes(
        content: *const c_char, content_bytes: usize,
        question: *const c_char, question_bytes: usize,
        criteria: *const DgCriteria, out_yes: *mut u8,
    ) -> i32;
    fn dg_is_yes_p(
        content: *const c_char, content_bytes: usize,
        question: *const c_char, question_bytes: usize,
        criteria: *const DgCriteria, out_p_yes: *mut f64,
    ) -> i32;
    fn dg_is_yes_at_threshold(
        content: *const c_char, content_bytes: usize,
        question: *const c_char, question_bytes: usize,
        criteria: *const DgCriteria, threshold: f64, out_yes: *mut u8,
    ) -> i32;
}

/// Ok(true / false) is an answer; Err is a failure, never disguised as "no".
fn is_yes(content: &str, question: &str) -> Result<bool, i32> {
    let mut yes = 0u8;
    let status = unsafe {
        dg_is_yes(
            content.as_ptr() as *const c_char, content.len(),
            question.as_ptr() as *const c_char, question.len(),
            std::ptr::null(), &mut yes,
        )
    };
    if status == 0 { Ok(yes == 1) } else { Err(status) }
}

/// Probability of yes, from 0.0 to 1.0.
fn is_yes_p(content: &str, question: &str) -> Result<f64, i32> {
    let mut p = 0f64;
    let status = unsafe {
        dg_is_yes_p(
            content.as_ptr() as *const c_char, content.len(),
            question.as_ptr() as *const c_char, question.len(),
            std::ptr::null(), &mut p,
        )
    };
    if status == 0 { Ok(p) } else { Err(status) }
}

/// Criteria and a stricter threshold.
fn is_yes_with(content: &str, question: &str, yes: &str, no: &str, threshold: f64)
    -> Result<bool, i32>
{
    let criteria = DgCriteria {
        yes: yes.as_ptr() as *const c_char, yes_bytes: yes.len(),
        no: no.as_ptr() as *const c_char, no_bytes: no.len(),
    };
    let mut out = 0u8;
    let status = unsafe {
        dg_is_yes_at_threshold(
            content.as_ptr() as *const c_char, content.len(),
            question.as_ptr() as *const c_char, question.len(),
            &criteria, threshold, &mut out,
        )
    };
    if status == 0 { Ok(out == 1) } else { Err(status) }
}

fn main() {
    let ticket = "Our whole warehouse can't print shipping labels and trucks leave in an hour.";
    match is_yes(ticket, "Is the customer describing an urgent problem?") {
        Ok(true) => println!("Page the on-call engineer"),
        Ok(false) => println!("Add to the normal queue"),
        Err(code) => eprintln!("Could not evaluate the question (status {code})"),
    }
    if let Ok(p) = is_yes_p(ticket, "Is the customer describing an urgent problem?") {
        println!("p_yes = {p:.3}");
    }
    let confident = is_yes_with(
        ticket,
        "Is the customer describing an urgent problem?",
        "Work is blocked and there is a deadline within hours.",
        "The problem is an inconvenience with no near deadline.",
        0.9,
    );
    println!("confident urgent: {confident:?}");
}
```

Wrap the `unsafe` calls once, as above, and the rest of your program sees ordinary `Result<bool, _>` values. The strings are borrowed for the duration of each call only.

## Go

Go calls the C interface through cgo, Go's built-in bridge to C, so building needs a C compiler: Xcode Command Line Tools on a Mac, gcc on Linux, or MinGW-w64 gcc on Windows. A working program is in [`examples/go`](examples/go); run it with `go run .` from that directory. It links the bundle under `released/<platform>/` by default. To use another bundle, set `CGO_LDFLAGS="-L/path/to/bundle -Wl,-rpath,/path/to/bundle"`. On Windows, put the bundle directory on `PATH` so the program finds the DLLs at run time.

The comment above `import "C"` tells cgo where the header and library are. The functions below it wrap each call once, so the rest of your program sees ordinary Go values and errors:

```go
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../code/include
#cgo LDFLAGS: -ldecisiongator
#cgo darwin LDFLAGS: -L${SRCDIR}/../../released/macos-arm64 -Wl,-rpath,${SRCDIR}/../../released/macos-arm64
#cgo linux LDFLAGS: -L${SRCDIR}/../../released/linux-x64 -Wl,-rpath,${SRCDIR}/../../released/linux-x64
#cgo windows LDFLAGS: -L${SRCDIR}/../../released/windows-x64
#include <stdlib.h>
#include "decisiongator.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"unsafe"
)

// Criteria spell out what counts as yes and what counts as no.
type Criteria struct{ Yes, No string }

// Ranked is one option's index into the caller's options and its probability.
type Ranked struct {
	Index int
	P     float64
}

// cstr borrows a Go string's bytes for the length of one call. The library
// takes explicit byte lengths, so no terminating NUL is needed.
func cstr(s string) (*C.char, C.size_t) {
	return (*C.char)(unsafe.Pointer(unsafe.StringData(s))), C.size_t(len(s))
}

func lastError(status C.dg_status) error {
	var required C.size_t
	C.dg_last_error(nil, 0, &required)
	if required == 0 {
		return fmt.Errorf("DecisionGator error %d", int(status))
	}
	buffer := make([]byte, int(required))
	C.dg_last_error((*C.char)(unsafe.Pointer(&buffer[0])), required, &required)
	return fmt.Errorf("DecisionGator error %d: %s", int(status), buffer[:len(buffer)-1])
}

// withCriteria passes nil for no criteria, or a C struct borrowing the strings.
func withCriteria(c *Criteria, call func(*C.dg_criteria) C.dg_status) C.dg_status {
	if c == nil {
		return call(nil)
	}
	var cc C.dg_criteria
	cc.yes, cc.yes_bytes = cstr(c.Yes)
	cc.no, cc.no_bytes = cstr(c.No)
	return call(&cc)
}

// IsYes answers a yes/no question about content. A nil error with false is a
// real "no"; failures are always returned as errors, never disguised as "no".
func IsYes(content, question string, criteria *Criteria, threshold float64) (bool, error) {
	text, textBytes := cstr(content)
	q, qBytes := cstr(question)
	var yes C.uint8_t
	status := withCriteria(criteria, func(c *C.dg_criteria) C.dg_status {
		return C.dg_is_yes_at_threshold(text, textBytes, q, qBytes, c, C.double(threshold), &yes)
	})
	if status != C.DG_OK {
		return false, lastError(status)
	}
	return yes == 1, nil
}

// IsYesP returns the probability of yes, from 0 to 1.
func IsYesP(content, question string, criteria *Criteria) (float64, error) {
	text, textBytes := cstr(content)
	q, qBytes := cstr(question)
	var p C.double
	status := withCriteria(criteria, func(c *C.dg_criteria) C.dg_status {
		return C.dg_is_yes_p(text, textBytes, q, qBytes, c, &p)
	})
	if status != C.DG_OK {
		return 0, lastError(status)
	}
	return float64(p), nil
}

// ChooseP ranks the options, best first, with probabilities that sum to one.
func ChooseP(content, question string, options []string, criteria *Criteria) ([]Ranked, error) {
	if len(options) == 0 {
		return nil, errors.New("no options")
	}
	// The option pointer array must live in C memory, so copy the options there.
	n := len(options)
	ptrs := (*[1 << 20]*C.char)(C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof((*C.char)(nil)))))[:n:n]
	sizes := make([]C.size_t, n)
	for i, option := range options {
		ptrs[i] = C.CString(option)
		sizes[i] = C.size_t(len(option))
	}
	defer func() {
		for _, p := range ptrs {
			C.free(unsafe.Pointer(p))
		}
		C.free(unsafe.Pointer(&ptrs[0]))
	}()
	text, textBytes := cstr(content)
	q, qBytes := cstr(question)
	index := make([]C.int32_t, n)
	prob := make([]C.double, n)
	status := withCriteria(criteria, func(c *C.dg_criteria) C.dg_status {
		return C.dg_choose_p(text, textBytes, q, qBytes, &ptrs[0], &sizes[0], C.size_t(n), c, &index[0], &prob[0])
	})
	if status != C.DG_OK {
		return nil, lastError(status)
	}
	ranked := make([]Ranked, n)
	for i := range ranked {
		ranked[i] = Ranked{int(index[i]), float64(prob[i])}
	}
	return ranked, nil
}

func main() {
	ticket := "Our whole warehouse can't print shipping labels and trucks leave in an hour."
	question := "Is the customer describing an urgent problem?"

	// A yes/no decision at the usual 0.5 cutoff.
	urgent, err := IsYes(ticket, question, nil, 0.5)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Could not evaluate the question:", err)
		os.Exit(1)
	}
	fmt.Println("urgent:", urgent)

	// The probability, so you can keep an uncertain range for a person.
	p, err := IsYesP(ticket, question, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("p_yes = %.3f\n", p)

	// Criteria and a stricter threshold.
	confident, err := IsYes(ticket, question, &Criteria{
		Yes: "Work is blocked and there is a deadline within hours.",
		No:  "The problem is an inconvenience with no near deadline.",
	}, 0.90)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("confident urgent:", confident)

	// Several options instead of yes or no.
	teams := []string{"billing", "technical support", "sales"}
	ranked, err := ChooseP("My card was charged twice for last month's invoice.",
		"Which team should handle this message?", teams, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, r := range ranked {
		fmt.Printf("%-18s %.3f\n", teams[r.Index], r.P)
	}
}
```

Strings are borrowed for the duration of each call only. `ChooseP` copies the options into C memory because cgo does not let C receive an array of Go pointers.

## Choosing a threshold

The default cutoff of 0.5 is a starting point, not a measured optimum. Collect a few dozen real examples from your application, run `is_yes_p` on them, and pick the cutoff that gives the mistakes you can tolerate. If it gets an example wrong, that example is training data: see [What to do when DecisionGator gives you a wrong answer](WRONG-ANSWER.md).
